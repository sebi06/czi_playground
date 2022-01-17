# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_metadata_old.py
# Version     : 0.0.5
# Author      : sebi06
# Date        : 17.01.2022
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
import os
import sys
from pathlib import Path
import xmltodict
from collections import Counter
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from aicspylibczi import CziFile
from tqdm.contrib.itertools import product
import pandas as pd
from czitools import misc
import numpy as np
import dateutil.parser as dt
import pydash
from typing import List, Dict, Tuple, Optional, Type, Any, Union


class CziMetadata:

    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = xmltodict.parse(czidoc.raw_metadata)

            # get directory, filename, SW version and acquisition data
            self.info = CziInfo(filename)

            # get dimensions
            self.pyczi_dims = czidoc.total_bounding_box
            self.dimstring = None
            self.dims_shape = None
            self.size = None

            # get the pixel typed for all channels
            self.pixeltypes = czidoc.pixel_types
            self.isRGB = False

            # determine pixel type for CZI array by reading XML metadata
            self.pixeltype = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["PixelType"]

            # check if CZI is a RGB file
            if self.pixeltype in ["Bgr24", "Bgr48", "Bgr96Float"]:
                self.isRGB = True

            # determine pixel type for CZI array
            self.npdtype, self.maxvalue = self.get_dtype_fromstring(self.pixeltype)

            # get the dimensions and order
            self.dims = CziDimensions(filename)

            # try to guess if the CZI is a mosaic file
            if self.dims.SizeM is None or self.dims.SizeM == 1:
                self.isMosaic = False
            else:
                self.isMosaic = True

            # get the bounding boxes
            self.bbox = CziBoundingBox(filename)

            # get information about channels
            self.channelinfo = CziChannelInfo(filename)

            # get scaling info
            self.scale = CziScaling(filename)

            # get objetive information
            self.objective = CziObjectives(filename)

            # get detector information
            self.detector = CziDetector(filename)

            # get detector information
            self.microscope = CziMicroscope(filename)

            # get information about sample carrier and wells etc.
            self.sample = CziSampleInfo(filename)

            # get additional metainformation
            self.add_metadata = CziAddMetaData(filename)

    # can be also used without creating an instance of the class
    @staticmethod
    def get_dtype_fromstring(pixeltype: str) -> Tuple[np.dtype, int]:

        dytpe = None

        if pixeltype == "gray16" or pixeltype == "Gray16":
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == "gray8" or pixeltype == "Gray8":
            dtype = np.dtype(np.uint8)
            maxvalue = 255
        if pixeltype == "bgr48" or pixeltype == "Bgr48":
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == "bgr24" or pixeltype == "Bgr24":
            dtype = np.dtype(np.uint8)
            maxvalue = 255
        if pixeltype == "bgr96float" or pixeltype == "Bgr96Float":
            dtype = np.dtype(np.uint16)
            maxvalue = 265535

        return dtype, maxvalue

    @staticmethod
    def get_dimorder(dimstring: str) -> Tuple[Dict, List, int]:
        """Get the order of dimensions from dimension string

        :param dimstring: string containing the dimensions
        :type dimstring: str
        :return: dims_dict - dictionary with the dimensions and its positions
        :rtype: dict
        :return: dimindex_list - list with indices of dimensions
        :rtype: list
        :return: numvalid_dims - number of valid dimensions
        :rtype: integer
        """

        dimindex_list = []
        dims = ["R", "I", "M", "H", "V", "B", "S", "T", "C", "Z", "Y", "X", "A"]
        dims_dict = {}

        # loop over all dimensions and find the index
        for d in dims:
            dims_dict[d] = dimstring.find(d)
            dimindex_list.append(dimstring.find(d))

        # check if a dimension really exists
        numvalid_dims = sum(i > 0 for i in dimindex_list)

        return dims_dict, dimindex_list, numvalid_dims

    @staticmethod
    def get_metadict(filename: str) -> Dict:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = xmltodict.parse(czidoc.raw_metadata)

        return md_dict


class CziDimensions:
    """
    Information official CZI Dimension Characters:
    - "X":"Width"
    - "Y":"Height"
    - "C":"Channel"
    - "Z":"Slice"        # depth
    - "T":"Time"
    - "R":"Rotation"
    - "S":"Scene"        # contiguous regions of interest in a mosaic image
    - "I":"Illumination" # direction
    - "B":"Block"        # acquisition
    - "M":"Mosaic"       # index of tile for compositing a scene
    - "H":"Phase"        # e.g. Airy detector fibers
    - "V":"View"         # e.g. for SPIM
    """

    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # get the dimensions
        self.SizeX = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeX"])
        self.SizeY = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeY"])

        # check C-Dimension
        try:
            self.SizeC = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeC"])
        except KeyError as e:
            self.SizeC = None

        # check Z-Dimension
        try:
            self.SizeZ = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeZ"])
        except KeyError as e:
            self.SizeZ = None

        # check T-Dimension
        try:
            self.SizeT = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeT"])
        except KeyError as e:
            self.SizeT = None

        # check M-Dimension
        try:
            self.SizeM = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeM"])
        except KeyError as e:
            self.SizeM = None

        # check B-Dimension
        try:
            self.SizeB = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeB"])
        except KeyError as e:
            self.SizeB = None

        # check S-Dimension
        try:
            self.SizeS = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeS"])
        except KeyError as e:
            self.SizeS = None

        # check H-Dimension
        try:
            self.SizeH = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeH"])
        except KeyError as e:
            self.SizeH = None

        # check I-Dimension
        try:
            self.SizeI = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeH"])
        except KeyError as e:
            self.SizeI = None

        # check R-Dimension
        try:
            self.SizeR = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeR"])
        except KeyError as e:
            self.SizeR = None


class CziBoundingBox:
    def __init__(self, filename: str) -> None:

        with pyczi.open_czi(filename) as czidoc:

            try:
                self.all_scenes = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.all_scenes = None
                print("Scenes Bounding rectangle not found.", e)

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                print("Total Bounding rectangle not found.", e)

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                print("Total Bounding Box not found.", e)


class CziChannelInfo:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # create empty lists for channel related information
        channels = []
        channels_names = []
        channels_colors = []
        channels_contrast = []
        channels_gamma = []

        try:
            sizeC = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeC"])
        except KeyError as e:
            sizeC = 1

        # in case of only one channel
        if sizeC == 1:
            # get name for dye
            try:
                channels.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["ShortName"])
            except KeyError as e:
                print("Channel shortname not found :", e)
                try:
                    channels.append(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["DyeName"])
                except KeyError as e:
                    print("Channel dye not found :", e)
                    channels.append("Dye-CH1")

            # get channel name
            try:
                channels_names.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Name"])
            except KeyError as e:
                try:
                    channels_names.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["@Name"])
                except KeyError as e:
                    print("Channel name found :", e)
                    channels_names.append("CH1")

            # get channel color
            try:
                channels_colors.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Color"])
            except KeyError as e:
                print("Channel color not found :", e)
                channels_colors.append("#80808000")

            # get contrast setting fro DisplaySetting
            try:
                low = np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Low"])
            except KeyError as e:
                low = 0.1
            try:
                high = np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["High"])
            except KeyError as e:
                high = 0.5

            channels_contrast.append([low, high])

            # get the gamma values
            try:
                channels_gamma.append(
                    np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Gamma"]))
            except KeyError as e:
                channels_gamma.append(0.85)

        # in case of two or more channels
        if sizeC > 1:
            # loop over all channels
            for ch in range(sizeC):
                # get name for dyes
                try:
                    channels.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["ShortName"])
                except KeyError as e:
                    print("Channel shortname not found :", e)
                    try:
                        channels.append(
                            md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch][
                                "DyeName"])
                    except KeyError as e:
                        print("Channel dye not found :", e)
                        channels.append("Dye-CH" + str(ch))

                # get channel names
                try:
                    channels_names.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Name"])
                except KeyError as e:
                    try:
                        channels_names.append(
                            md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["@Name"])
                    except KeyError as e:
                        print("Channel name not found :", e)
                        channels_names.append("CH" + str(ch))

                # get channel colors
                try:
                    channels_colors.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Color"])
                except KeyError as e:
                    print("Channel color not found :", e)
                    # use grayscale instead
                    channels_colors.append("80808000")

                # get contrast setting fro DisplaySetting
                try:
                    low = np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Low"])
                except KeyError as e:
                    low = 0.0
                try:
                    high = np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["High"])
                except KeyError as e:
                    high = 0.5

                channels_contrast.append([low, high])

                # get the gamma values
                try:
                    channels_gamma.append(np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Gamma"]))
                except KeyError as e:
                    channels_gamma.append(0.85)

        # write channels information (as lists) into metadata dictionary
        self.shortnames = channels
        self.names = channels_names
        self.colors = channels_colors
        self.clims = channels_contrast
        self.gamma = channels_gamma


class CziScaling:
    def __init__(self, filename: str, dim2none: bool = True) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # get the XY scaling information
        try:
            self.X = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]) * 1000000
            self.Y = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][1]["Value"]) * 1000000
            self.X = np.round(self.X, 3)
            self.Y = np.round(self.Y, 3)
        except (KeyError, TypeError) as e:
            print("Error extracting XY Scale  :", e)
            self.X = 1.0
            self.Y = 1.0

        try:
            self.XUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["DefaultUnitFormat"]
            self.YUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][1]["DefaultUnitFormat"]
        except (KeyError, TypeError) as e:
            print("Error extracting XY ScaleUnit :", e)
            self.XUnit = None
            self.YUnit = None

        # get the Z scaling information
        try:
            self.Z = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][2]["Value"]) * 1000000
            self.Z = np.round(self.Z, 3)
            # additional check for faulty z-scaling
            if self.Z == 0.0:
                self.Z = 1.0
            try:
                self.ZUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][2]["DefaultUnitFormat"]
            except (IndexError, KeyError, TypeError) as e:
                print("Error extracting Z ScaleUnit :", e)
                self.ZUnit = self.XUnit
        except (IndexError, KeyError, TypeError) as e:
            print("Error extracting Z Scale  :", e)
            # set to isotropic scaling if it was single plane only
            self.Z = self.X
            self.ZUnit = self.XUnit

        # convert scale unit to avoid encoding problems
        if self.XUnit == "µm":
            self.XUnit = "micron"
        if self.YUnit == "µm":
            self.YUnit = "micron"
        if self.ZUnit == "µm":
            self.ZUnit = "micron"

        # get scaling ratio
        self.ratio = self.get_scale_ratio(scalex=self.X,
                                          scaley=self.Y,
                                          scalez=self.Z)

    @staticmethod
    def get_scale_ratio(scalex: float = 1.0,
                        scaley: float = 1.0,
                        scalez: float = 1.0) -> Dict:

        # set default scale factor to 1.0
        scale_ratio = {"xy": 1.0,
                       "zx": 1.0
                       }
        try:
            # get the factor between XY scaling
            scale_ratio["xy"] = np.round(scalex / scaley, 3)
            # get the scalefactor between XZ scaling
            scale_ratio["zx"] = np.round(scalez / scalex, 3)
        except TypeError as e:
            print(e, "Using defaults = 1.0")

        return scale_ratio


class CziInfo:
    def __init__(self, filename: str) -> None:

        # get directory and filename etc.
        self.dirname = os.path.dirname(filename)
        self.filename = os.path.basename(filename)

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # get acquisition data and SW version
        try:
            self.software_name = md_dict["ImageDocument"]["Metadata"]["Information"]["Application"]["Name"]
            self.software_version = md_dict["ImageDocument"]["Metadata"]["Information"]["Application"]["Version"]
        except KeyError as e:
            print("Key not found:", e)
            self.software_name = None
            self.software_version = None

        try:
            self.acquisition_date = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["AcquisitionDateAndTime"]
        except KeyError as e:
            print("Key not found:", e)
            self.acquisition_date = None


class CziObjectives:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        self.NA = []
        self.mag = []
        self.ID = []
        self.name = []
        self.immersion = []
        self.tubelensmag = []
        self.nominalmag = []

        # check if Instrument metadata actually exist
        if pydash.objects.has(md_dict, ["ImageDocument", "Metadata", "Information", "Instrument", "Objectives"]):
            # get objective data
            try:
                if isinstance(md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"], list):
                    num_obj = len(md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'])
                else:
                    num_obj = 1
            except KeyError as e:
                num_obj = 0  # no objective found

            # if there is only one objective found
            if num_obj == 1:
                try:
                    self.name.append(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Name'])
                except (KeyError, TypeError) as e:
                    print('No Objective Name :', e)
                    self.name.append(None)

                try:
                    self.immersion = md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Immersion']
                except (KeyError, TypeError) as e:
                    print('No Objective Immersion :', e)
                    self.immersion = None

                try:
                    self.NA = np.float(md_dict['ImageDocument']['Metadata']['Information']
                                       ['Instrument']['Objectives']['Objective']['LensNA'])
                except (KeyError, TypeError) as e:
                    print('No Objective NA :', e)
                    self.NA = None

                try:
                    self.ID = md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Id']
                except (KeyError, TypeError) as e:
                    print('No Objective ID :', e)
                    self.ID = None

                try:
                    self.tubelensmag = np.float(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['TubeLenses']['TubeLens']['Magnification'])
                except (KeyError, TypeError) as e:
                    print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                    self.tubelensmag = 1.0

                try:
                    self.nominalmag = np.float(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][
                            'NominalMagnification'])
                except (KeyError, TypeError) as e:
                    print('No Nominal Mag.:', e, 'Using Default Value = 1.0.')
                    self.nominalmag = 1.0

                try:
                    if self.tubelensmag is not None:
                        self.mag = self.nominalmag * self.tubelensmag
                    if self.tubelensmag is None:
                        print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                        self.mag = self.nominalmag * 1.0

                except (KeyError, TypeError) as e:
                    print('No Objective Magnification :', e)
                    self.mag = None

            if num_obj > 1:
                for o in range(num_obj):

                    try:
                        self.name.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o][
                                'Name'])
                    except KeyError as e:
                        print('No Objective Name :', e)
                        self.name.append(None)

                    try:
                        self.immersion.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o][
                                'Immersion'])
                    except KeyError as e:
                        print('No Objective Immersion :', e)
                        self.immersion.append(None)

                    try:
                        self.NA.append(np.float(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o][
                                'LensNA']))
                    except KeyError as e:
                        print('No Objective NA :', e)
                        self.NA.append(None)

                    try:
                        self.ID.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o][
                                'Id'])
                    except KeyError as e:
                        print('No Objective ID :', e)
                        self.ID.append(None)

                    try:
                        self.tubelensmag.append(np.float(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['TubeLenses']['TubeLens'][o][
                                'Magnification']))
                    except KeyError as e:
                        print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                        self.tubelensmag.append(1.0)

                    try:
                        self.nominalmag.append(np.float(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o][
                                'NominalMagnification']))
                    except KeyError as e:
                        print('No Nominal Mag. :', e, 'Using Default Value = 1.0.')
                        self.nominalmag.append(1.0)

                    try:
                        if self.tubelensmag is not None:
                            self.mag.append(self.nominalmag[o] * self.tubelensmag[o])
                        if self.tubelensmag is None:
                            print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                            self.mag.append(self.nominalmag[o] * 1.0)

                    except KeyError as e:
                        print('No Objective Magnification :', e)
                        self.mag.append(None)


class CziDetector:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # get detector information
        self.model = []
        self.name = []
        self.ID = []
        self.modeltype = []
        self.instrumentID = []

        # check if there are any detector entries inside the dictionary
        if pydash.objects.has(md_dict, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Detectors']):

            if isinstance(md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'], list):
                num_detectors = len(
                    md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'])
            else:
                num_detectors = 1

            # if there is only one detector found
            if num_detectors == 1:

                # check for detector ID
                try:
                    self.ID.append(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Id'])
                except KeyError as e:
                    print('DetectorID not found :', e)
                    self.ID.append(None)

                # check for detector Name
                try:
                    self.name.append(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Name'])
                except KeyError as e:
                    print('DetectorName not found :', e)
                    self.name.append(None)

                # check for detector model
                try:
                    self.model.append(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][
                            'Manufacturer']['Model'])
                except KeyError as e:
                    print('DetectorModel not found :', e)
                    self.model.append(None)

                # check for detector modeltype
                try:
                    self.modeltype.append(
                        md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Type'])
                except KeyError as e:
                    print('DetectorType not found :', e)
                    self.modeltype.append(None)

            if num_detectors > 1:
                for d in range(num_detectors):

                    # check for detector ID
                    try:
                        self.ID.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d][
                                'Id'])
                    except KeyError as e:
                        print('DetectorID not found :', e)
                        self.ID.append(None)

                    # check for detector Name
                    try:
                        self.name.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d][
                                'Name'])
                    except KeyError as e:
                        print('DetectorName not found :', e)
                        self.name.append(None)

                    # check for detector model
                    try:
                        self.model.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d][
                                'Manufacturer']['Model'])
                    except KeyError as e:
                        print('DetectorModel not found :', e)
                        self.model.append(None)

                    # check for detector modeltype
                    try:
                        self.modeltype.append(
                            md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d][
                                'Type'])
                    except KeyError as e:
                        print('DetectorType not found :', e)
                        self.modeltype.append(None)


class CziMicroscope:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        self.ID = None
        self.Name = None

        # check if there are any microscope entry inside the dictionary
        if pydash.objects.has(md_dict, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Microscopes']):

            # check for detector ID
            try:
                self.ID = md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope'][
                    'Id']
            except KeyError as e:
                try:
                    self.ID = md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope'][
                        '@Id']
                except KeyError as e:
                    print('Microscope ID not found :', e)
                    self.ID = None

            # check for microscope system name
            try:
                self.Name = md_dict['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope'][
                    'System']
            except KeyError as e:
                print('Microscope System Name not found :', e)
                self.Name = None


class CziSampleInfo:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        # check for well information
        self.well_array_names = []
        self.well_indices = []
        self.well_position_names = []
        self.well_colID = []
        self.well_rowID = []
        self.well_counter = []
        self.scene_stageX = []
        self.scene_stageY = []

        try:
            # get S-Dimension
            sizeS = np.int(md_dict['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
            print('Trying to extract Scene and Well information if existing ...')

            # extract well information from the dictionary
            allscenes = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']

            # loop over all detected scenes
            for s in range(sizeS):

                if sizeS == 1:
                    well = allscenes
                    try:
                        self.well_array_names.append(allscenes['ArrayName'])
                    except KeyError as e:
                        try:
                            self.well_array_names.append(well['Name'])
                        except KeyError as e:
                            # print('Well Name not found :', e)
                            try:
                                self.well_array_names.append(well['@Name'])
                            except KeyError as e:
                                # print('Well @Name not found :', e)
                                print('Well Name not found :', e, 'Using A1 instead')
                                self.well_array_names.append('A1')

                    try:
                        self.well_indices.append(allscenes['Index'])
                    except KeyError as e:
                        try:
                            self.well_indices.append(allscenes['@Index'])
                        except KeyError as e:
                            print('Well Index not found :', e)
                            self.well_indices.append(1)

                    try:
                        self.well_position_names.append(allscenes['Name'])
                    except KeyError as e:
                        try:
                            self.well_position_names.append(allscenes['@Name'])
                        except KeyError as e:
                            print('Well Position Names not found :', e)
                            self.well_position_names.append('P1')

                    try:
                        self.well_colID.append(np.int(allscenes['Shape']['ColumnIndex']))
                    except KeyError as e:
                        print('Well ColumnIDs not found :', e)
                        self.well_colID.append(0)

                    try:
                        self.well_rowID.append(np.int(allscenes['Shape']['RowIndex']))
                    except KeyError as e:
                        print('Well RowIDs not found :', e)
                        self.well_rowID.append(0)

                    try:
                        # count the content of the list, e.g. how many time a certain well was detected
                        self.well_counter = Counter(self.well_array_names)
                    except KeyError:
                        self.well_counter.append(Counter({'A1': 1}))

                    try:
                        # get the SceneCenter Position
                        sx = allscenes['CenterPosition'].split(',')[0]
                        sy = allscenes['CenterPosition'].split(',')[1]
                        self.scene_stageX.append(np.double(sx))
                        self.scene_stageY.append(np.double(sy))
                    except (TypeError, KeyError) as e:
                        print('Stage Positions XY not found :', e)
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                if sizeS > 1:
                    try:
                        well = allscenes[s]
                        self.well_array_names.append(well['ArrayName'])
                    except KeyError as e:
                        try:
                            self.well_array_names.append(well['Name'])
                        except KeyError as e:
                            # print('Well Name not found :', e)
                            try:
                                self.well_array_names.append(well['@Name'])
                            except KeyError as e:
                                # print('Well @Name not found :', e)
                                print('Well Name not found. Using A1 instead')
                                self.well_array_names.append('A1')

                    # get the well information
                    try:
                        self.well_indices.append(well['Index'])
                    except KeyError as e:
                        try:
                            self.well_indices.append(well['@Index'])
                        except KeyError as e:
                            print('Well Index not found :', e)
                            self.well_indices.append(None)
                    try:
                        self.well_position_names.append(well['Name'])
                    except KeyError as e:
                        try:
                            self.well_position_names.append(well['@Name'])
                        except KeyError as e:
                            print('Well Position Names not found :', e)
                            self.well_position_names.append(None)

                    try:
                        self.well_colID.append(np.int(well['Shape']['ColumnIndex']))
                    except KeyError as e:
                        print('Well ColumnIDs not found :', e)
                        self.well_colID.append(None)

                    try:
                        self.well_rowID.append(np.int(well['Shape']['RowIndex']))
                    except KeyError as e:
                        print('Well RowIDs not found :', e)
                        self.well_rowID.append(None)

                    # count the content of the list, e.g. how many time a certain well was detected
                    self.well_counter = Counter(self.well_array_names)

                    # try:
                    if isinstance(allscenes, list):
                        try:
                            # get the SceneCenter Position
                            sx = allscenes[s]['CenterPosition'].split(',')[0]
                            sy = allscenes[s]['CenterPosition'].split(',')[1]
                            self.scene_stageX.append(np.double(sx))
                            self.scene_stageY.append(np.double(sy))
                        except KeyError as e:
                            print('Stage Positions XY not found :', e)
                            self.scene_stageX.append(0.0)
                            self.scene_stageY.append(0.0)
                    if not isinstance(allscenes, list):
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                # count the number of different wells
                self.number_wells = len(self.well_counter.keys())

        except (KeyError, TypeError) as e:
            print('No valid Scene or Well information found:', e)


class CziAddMetaData:
    def __init__(self, filename: str) -> None:

        # get the metadata as a dictionary
        md_dict = CziMetadata.get_metadict(filename)

        try:
            self.experiment = md_dict['ImageDocument']['Metadata']['Experiment']
        except KeyError as e:
            print('Key not found :', e)
            self.experiment = None

        try:
            self.hardwaresetting = md_dict['ImageDocument']['Metadata']['HardwareSetting']
        except KeyError as e:
            print('Key not found :', e)
            self.hardwaresetting = None

        try:
            self.customattributes = md_dict['ImageDocument']['Metadata']['CustomAttributes']
        except KeyError as e:
            print('Key not found :', e)
            self.customattributes = None

        try:
            self.displaysetting = md_dict['ImageDocument']['Metadata']['DisplaySetting']
        except KeyError as e:
            print('Key not found :', e)
            self.displaysetting = None

        try:
            self.layers = md_dict['ImageDocument']['Metadata']['Layers']
        except KeyError as e:
            print('Key not found :', e)
            self.layers = None


class CziScene:
    def __init__(self, filename: str, sceneindex: int) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = xmltodict.parse(czidoc.raw_metadata)

            self.bbox = czidoc.scenes_bounding_rectangle[sceneindex]
            self.xstart = self.bbox.x
            self.ystart = self.bbox.y
            self.width = self.bbox.w
            self.height = self.bbox.h
            self.index = sceneindex

        # TODO : And scene dimensions to CziScene class


def create_mdict_complete(metadata: Union[str, CziMetadata], sort: bool = True) -> Dict:
    """
    Created a metadata dictionary. Accepts a filename of a CZI file or
    a CziMetadata class

    Args:
        metadata: filename or CziMetadata class
        sort: sort the dictionary

    Returns: dictionary with the metadata

    """
    if isinstance(metadata, str):
        # get the metadata as a dictionary from filename
        metadata = CziMetadata(metadata)

    # create a dictionary with the metadata

    # Attemtion: the list of keys is not complete!
    md_dict = {'Directory': metadata.info.dirname,
               'Filename': metadata.info.filename,
               'AcqDate': metadata.info.acquisition_date,
               'SW-Name': metadata.info.software_name,
               'SW-Version': metadata.info.software_version,
               'czi_dims': metadata.dimstring,
               'czi_dims_shape': metadata.dims_shape,
               'czi_size': metadata.size,
               # 'dim_order': metadata.dim_order,
               # 'dim_index': metadata.dim_index,
               # 'dim_valid': metadata.dim_valid,
               'SizeX': metadata.dims.SizeX,
               'SizeY': metadata.dims.SizeY,
               'SizeZ': metadata.dims.SizeZ,
               'SizeC': metadata.dims.SizeC,
               'SizeT': metadata.dims.SizeT,
               'SizeS': metadata.dims.SizeS,
               'SizeB': metadata.dims.SizeB,
               'SizeM': metadata.dims.SizeM,
               'SizeH': metadata.dims.SizeH,
               'SizeI': metadata.dims.SizeI,
               'isRGB': metadata.isRGB,
               'isMosaic': metadata.isMosaic,
               'ObjNA': metadata.objective.NA,
               'ObjMag': metadata.objective.mag,
               'ObjID': metadata.objective.ID,
               'ObjName': metadata.objective.name,
               'ObjImmersion': metadata.objective.immersion,
               'TubelensMag': metadata.objective.tubelensmag,
               'ObjNominalMag': metadata.objective.nominalmag,
               'XScale': metadata.scale.X,
               'YScale': metadata.scale.Y,
               'ZScale': metadata.scale.Z,
               'XScaleUnit': metadata.scale.XUnit,
               'YScaleUnit': metadata.scale.YUnit,
               'ZScaleUnit': metadata.scale.ZUnit,
               'scale_ratio': metadata.scale.ratio,
               'DetectorModel': metadata.detector.model,
               'DetectorName': metadata.detector.name,
               'DetectorID': metadata.detector.ID,
               'DetectorType': metadata.detector.modeltype,
               'InstrumentID': metadata.detector.instrumentID,
               'ChannelsNames': metadata.channelinfo.names,
               'ChannelShortNames': metadata.channelinfo.shortnames,
               'ChannelColors': metadata.channelinfo.colors,
               'bbox_all_scenes': metadata.bbox.all_scenes,
               'WellArrayNames': metadata.sample.well_array_names,
               'WellIndicies': metadata.sample.well_indices,
               'WellPositionNames': metadata.sample.well_position_names,
               'WellRowID': metadata.sample.well_rowID,
               'WellColumnID': metadata.sample.well_colID,
               'WellCounter': metadata.sample.well_counter,
               'SceneCenterStageX': metadata.sample.scene_stageX,
               'SceneCenterStageY': metadata.sample.scene_stageX
               }

    # check fro extra entries when reading mosaic file with a scale factor
    if hasattr(metadata.dims, "SizeX_sf"):
        md_dict['SizeX sf'] = metadata.dims.SizeX_sf
        md_dict['SizeY sf'] = metadata.dims.SizeY_sf
        md_dict['XScale sf'] = metadata.scale.X_sf
        md_dict['YScale sf'] = metadata.scale.Y_sf
        md_dict['ratio sf'] = metadata.scale.ratio_sf
        md_dict['scalefactorXY'] = metadata.scale.scalefactorXY

    # add info for bounding boxes
    md_dict['bbox_all_scenes'] = metadata.bbox.all_scenes
    md_dict['bbox_total_rect'] = metadata.bbox.total_rect
    md_dict['bbox_total'] = metadata.bbox.total_bounding_box

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict
