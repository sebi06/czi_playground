# -*- coding: utf-8 -*-

#################################################################
# File        : imgfileutils.py
# Version     : 1.3
# Author      : czsrh
# Date        : 12.07.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################


import czifile as zis
from apeer_ometiff_library import omexmlClass
import os
from pathlib import Path
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xmltodict
import numpy as np
from collections import Counter
from lxml import etree as ET
import time
import re
import sys
from aicsimageio import AICSImage, imread, imread_dask
from aicsimageio.writers import ome_tiff_writer
from aicspylibczi import CziFile
import dask.array as da
import pandas as pd
import tifffile

try:
    import javabridge as jv
    import bioformats
except (ImportError, ModuleNotFoundError) as error:
    # Output expected ImportErrors.
    print(error.__class__.__name__ + ": " + error.msg)
    print('Python-BioFormats cannot be used')

try:
    import ipywidgets as widgets
except ModuleNotFoundError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import napari
except ModuleNotFoundError as error:
    print(error.__class__.__name__ + ": " + error.msg)


def get_imgtype(imagefile):
    """Returns the type of the image based on the file extension - no magic

    :param imagefile: filename of the image
    :type imagefile: str
    :return: string specifying the image type
    :rtype: str
    """

    imgtype = None

    if imagefile.lower().endswith('.ome.tiff') or imagefile.lower().endswith('.ome.tif'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'ometiff'

    elif imagefile.lower().endswith('.tiff') or imagefile.lower().endswith('.tif'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'tiff'

    elif imagefile.lower().endswith('.czi'):
        # it is on CZI based on the file extension ... :-)
        imgtype = 'czi'

    elif imagefile.lower().endswith('.png'):
        # it is on CZI based on the file extension ... :-)
        imgtype = 'png'

    elif imagefile.lower().endswith('.jpg') or imagefile.lower().endswith('.jpeg'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'jpg'

    return imgtype


def create_metadata_dict():
    """A Python dictionary will be created to hold the relevant metadata.

    :return: dictionary with keys for the relevant metadata
    :rtype: dict
    """

    metadata = {'Directory': None,
                'Filename': None,
                'Extension': None,
                'ImageType': None,
                'AcqDate': None,
                'TotalSeries': None,
                'SizeX': None,
                'SizeY': None,
                'SizeZ': 1,
                'SizeC': 1,
                'SizeT': 1,
                'SizeS': 1,
                'SizeB': 1,
                'SizeM': 1,
                'SizeM': 1,
                'Sizes BF': None,
                # 'DimOrder BF': None,
                # 'DimOrder BF Array': None,
                # 'Axes_czifile': None,
                # 'Shape_czifile': None,
                'isRGB': False,
                'ObjNA': None,
                'ObjMag': None,
                'ObjID': None,
                'ObjName': None,
                'ObjImmersion': None,
                'XScale': None,
                'YScale': None,
                'ZScale': None,
                'XScaleUnit': None,
                'YScaleUnit': None,
                'ZScaleUnit': None,
                'DetectorModel': [],
                'DetectorName': [],
                'DetectorID': [],
                'DetectorType': [],
                'InstrumentID': [],
                'Channels': [],
                'ChannelNames': [],
                'ChannelColors': [],
                'ImageIDs': [],
                'NumPy.dtype': None
                }

    return metadata


def get_metadata(imagefile,
                 omeseries=0,
                 round_values=False):
    """Returns a dictionary with metadata depending on the image type.
    Only CZI and OME-TIFF are currently supported.

    :param imagefile: filename of the image
    :type imagefile: str
    :param omeseries: series of OME-TIFF file, , defaults to 0
    :type omeseries: int, optional
    :param round_values: option to round some values, defaults to TrueFalse
    :type round_values: bool, optional
    :return: metadata - dict with the metainformation
    :rtype: dict
    :return: additional_mdczi - dict with additional the metainformation for CZI only
    :rtype: dict
    """

    # get the image type
    imgtype = get_imgtype(imagefile)
    print('Detected Image Type (based on extension): ', imgtype)

    md = {}
    additional_md = {}

    if imgtype == 'ometiff':

        # parse the OME-XML and return the metadata dictionary and additional info
        md = get_metadata_ometiff(imagefile, series=omeseries)

    elif imgtype == 'czi':

        # parse the CZI metadata return the metadata dictionary and additional info
        md = get_metadata_czi(imagefile, dim2none=False)
        additional_md = get_additional_metadata_czi(imagefile)

    # TODO - Remove this when issue is fixed
    if round_values:
        # temporary workaround for slider / floating point issue in Napari viewer
        # https://forum.image.sc/t/problem-with-dimension-slider-when-adding-array-as-new-layer-for-ome-tiff/39092/2?u=sebi06

        md['XScale'] = np.round(md['XScale'], 3)
        md['YScale'] = np.round(md['YScale'], 3)
        md['ZScale'] = np.round(md['ZScale'], 3)
    else:
        # no metadate will be returned
        print('No metadata will be returned.')

    return md, additional_md


def get_metadata_ometiff(filename, series=0):
    """Returns a dictionary with OME-TIFF metadata.

    :param filename: filename of the OME-TIFF image
    :type filename: str
    :param series: Image Series, defaults to 0
    :type series: int, optional
    :return: dictionary with the relevant OME-TIFF metainformation
    :rtype: dict
    """

    with tifffile.TiffFile(filename) as tif:
        try:
            # get OME-XML metadata as string the old way
            omexml_string = tif[0].image_description.decode('utf-8')
        except TypeError as error:
            omexml_string = tif.ome_metadata

    # get the OME-XML using the apeer-ometiff-library
    omemd = omexmlClass.OMEXML(omexml_string)

    # create dictionary for metadata and get OME-XML data
    metadata = create_metadata_dict()

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['Extension'] = 'ome.tiff'
    metadata['ImageType'] = 'ometiff'
    metadata['AcqDate'] = omemd.image(series).AcquisitionDate
    metadata['Name'] = omemd.image(series).Name

    # get image dimensions TZCXY
    metadata['SizeT'] = omemd.image(series).Pixels.SizeT
    metadata['SizeZ'] = omemd.image(series).Pixels.SizeZ
    metadata['SizeC'] = omemd.image(series).Pixels.SizeC
    metadata['SizeX'] = omemd.image(series).Pixels.SizeX
    metadata['SizeY'] = omemd.image(series).Pixels.SizeY

    # get number of image series
    metadata['TotalSeries'] = omemd.get_image_count()
    metadata['Sizes BF'] = [metadata['TotalSeries'],
                            metadata['SizeT'],
                            metadata['SizeZ'],
                            metadata['SizeC'],
                            metadata['SizeY'],
                            metadata['SizeX']]

    # get dimension order
    metadata['DimOrder BF'] = omemd.image(series).Pixels.DimensionOrder

    # reverse the order to reflect later the array shape
    metadata['DimOrder BF Array'] = metadata['DimOrder BF'][::-1]

    # get the scaling
    metadata['XScale'] = omemd.image(series).Pixels.PhysicalSizeX
    metadata['XScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeXUnit
    metadata['YScale'] = omemd.image(series).Pixels.PhysicalSizeY
    metadata['YScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeYUnit
    metadata['ZScale'] = omemd.image(series).Pixels.PhysicalSizeZ
    metadata['ZScaleUnit'] = omemd.image(series).Pixels.PhysicalSizeZUnit

    # get all image IDs
    for i in range(omemd.get_image_count()):
        metadata['ImageIDs'].append(i)

    # get information about the instrument and objective
    try:
        metadata['InstrumentID'] = omemd.instrument(series).get_ID()
    except KeyError as e:
        print('Key not found:', e)
        metadata['InstrumentID'] = None

    try:
        metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Model()
        metadata['DetectorID'] = omemd.instrument(series).Detector.get_ID()
        metadata['DetectorModel'] = omemd.instrument(series).Detector.get_Type()
    except KeyError as e:
        print('Key not found:', e)
        metadata['DetectorModel'] = None
        metadata['DetectorID'] = None
        metadata['DetectorModel'] = None

    try:
        metadata['ObjNA'] = omemd.instrument(series).Objective.get_LensNA()
        metadata['ObjID'] = omemd.instrument(series).Objective.get_ID()
        metadata['ObjMag'] = omemd.instrument(series).Objective.get_NominalMagnification()
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjNA'] = None
        metadata['ObjID'] = None
        metadata['ObjMag'] = None

    # get channel names
    for c in range(metadata['SizeC']):
        metadata['Channels'].append(omemd.image(series).Pixels.Channel(c).Name)

    # add axes and shape information using aicsimageio package
    ometiff_aics = AICSImage(filename)
    metadata['Axes_aics'] = ometiff_aics.dims
    metadata['Shape_aics'] = ometiff_aics.shape
    metadata['SizeX_aics'] = ometiff_aics.size_x
    metadata['SizeY_aics'] = ometiff_aics.size_y
    metadata['SizeC_aics'] = ometiff_aics.size_c
    metadata['SizeZ_aics'] = ometiff_aics.size_t
    metadata['SizeT_aics'] = ometiff_aics.size_t
    metadata['SizeS_aics'] = ometiff_aics.size_s

    # close AICSImage object
    ometiff_aics.close()

    # check for None inside Scaling to avoid issues later one ...
    metadata = checkmdscale_none(metadata,
                                 tocheck=['XScale', 'YScale', 'ZScale'],
                                 replace=[1.0, 1.0, 1.0])

    return metadata


def checkmdscale_none(md, tocheck=['ZScale'], replace=[1.0]):
    """Check scaling entries for None to avoid issues later on

    :param md: original metadata
    :type md: dict
    :param tocheck: list with entries to check for None, defaults to ['ZScale']
    :type tocheck: list, optional
    :param replace: list with values replacing the None, defaults to [1.0]
    :type replace: list, optional
    :return: modified metadata where None entries where replaces by
    :rtype: [type]
    """
    for tc, rv in zip(tocheck, replace):
        if md[tc] is None:
            md[tc] = rv

    return md


def get_metadata_czi(filename, dim2none=False):
    """
    Returns a dictionary with CZI metadata.

    Information CZI Dimension Characters:
    - '0': 'Sample',  # e.g. RGBA
    - 'X': 'Width',
    - 'Y': 'Height',
    - 'C': 'Channel',
    - 'Z': 'Slice',  # depth
    - 'T': 'Time',
    - 'R': 'Rotation',
    - 'S': 'Scene',  # contiguous regions of interest in a mosaic image
    - 'I': 'Illumination',  # direction
    - 'B': 'Block',  # acquisition
    - 'M': 'Mosaic',  # index of tile for compositing a scene
    - 'H': 'Phase',  # e.g. Airy detector fibers
    - 'V': 'View',  # e.g. for SPIM

    :param filename: filename of the CZI image
    :type filename: str
    :param dim2none: option to set non-existing dimension to None, defaults to False
    :type dim2none: bool, optional
    :return: metadata - dictionary with the relevant CZI metainformation
    :rtype: dict
    """

    # get CZI object
    czi = zis.CziFile(filename)

    # parse the XML into a dictionary
    metadatadict_czi = czi.metadata(raw=False)

    # initialize metadata dictionary
    metadata = create_metadata_dict()

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['Extension'] = 'czi'
    metadata['ImageType'] = 'czi'

    # add axes and shape information using czifile package
    metadata['Axes_czifile'] = czi.axes
    metadata['Shape_czifile'] = czi.shape

    # add axes and shape information using aicsimageio package
    czi_aics = AICSImage(filename)
    metadata['Axes_aics'] = czi_aics.dims
    metadata['Shape_aics'] = czi_aics.shape
    metadata['SizeX_aics'] = czi_aics.size_x
    metadata['SizeY_aics'] = czi_aics.size_y
    metadata['SizeC_aics'] = czi_aics.size_c
    metadata['SizeZ_aics'] = czi_aics.size_t
    metadata['SizeT_aics'] = czi_aics.size_t
    metadata['SizeS_aics'] = czi_aics.size_s

    # get additional data by using pylibczi directly
    # Get the shape of the data, the coordinate pairs are (start index, size)
    aics_czi = CziFile(filename)
    metadata['dims_aicspylibczi'] = aics_czi.dims_shape()[0]
    metadata['dimorder_aicspylibczi'] = aics_czi.dims
    metadata['size_aicspylibczi'] = aics_czi.size
    metadata['czi_ismosaic'] = aics_czi.is_mosaic()

    # determine pixel type for CZI array
    metadata['NumPy.dtype'] = czi.dtype

    # check if the CZI image is an RGB image depending
    # on the last dimension entry of axes
    if czi.axes[-1] == 3:
        metadata['isRGB'] = True

    try:
        metadata['PixelType'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['PixelType']
    except KeyError as e:
        print('Key not found:', e)
        metadata['PixelType'] = None

    metadata['SizeX'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeX'])
    metadata['SizeY'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeY'])

    try:
        metadata['SizeZ'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    # create empty lists for channel related information
    channels = []
    channels_names = []
    channels_colors = []

    # in case of only one channel
    if metadata['SizeC'] == 1:
        # get name for dye
        try:
            channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                            ['Channels']['Channel']['ShortName'])
        except KeyError as e:
            print('Exception:', e)
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                ['Channels']['Channel']['DyeName'])
            except KeyError as e:
                print('Exception:', e)
                channels.append('Dye-CH1')

        # get channel name
        try:
            channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                  ['Channels']['Channel']['Name'])
        except KeyError as e:
            print('Exception:', e)
            channels_names.append['CH1']

        # get channel color
        try:
            channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                   ['Channels']['Channel']['Color'])
        except KeyError as e:
            print('Exception:', e)
            channels_colors.append('80808000')

    # in case of two or more channels
    if metadata['SizeC'] > 1:
        # loop over all channels
        for ch in range(metadata['SizeC']):
            # get name for dyes
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                ['Channels']['Channel'][ch]['ShortName'])
            except KeyError as e:
                print('Exception:', e)
                try:
                    channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                    ['Channels']['Channel'][ch]['DyeName'])
                except KeyError as e:
                    print('Exception:', e)
                    channels.append('Dye-CH' + str(ch))

            # get channel names
            try:
                channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                      ['Channels']['Channel'][ch]['Name'])
            except KeyError as e:
                print('Exception:', e)
                channels_names.append('CH' + str(ch))

            # get channel colors
            try:
                channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                       ['Channels']['Channel'][ch]['Color'])
            except KeyError as e:
                print('Exception:', e)
                # use grayscale instead
                channels_colors.append('80808000')

    # write channels information (as lists) into metadata dictionary
    metadata['Channels'] = channels
    metadata['ChannelNames'] = channels_names
    metadata['ChannelColors'] = channels_colors

    try:
        metadata['SizeT'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeT'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeM'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeB'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['SizeH'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeH'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeH'] = None
        if not dim2none:
            metadata['SizeH'] = 1

    try:
        metadata['SizeI'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeI'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeI'] = None
        if not dim2none:
            metadata['SizeI'] = 1

    try:
        metadata['SizeV'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeV'])
    except Exception as e:
        # print('Exception:', e)
        if dim2none:
            metadatada['SizeV'] = None
        if not dim2none:
            metadata['SizeV'] = 1

    # get the scaling information
    try:
        # metadata['Scaling'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']
        metadata['XScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['YScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        # metadata['XScale'] = np.round(metadata['XScale'], 3)
        # metadata['YScale'] = np.round(metadata['YScale'], 3)
        try:
            metadata['XScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
            metadata['YScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
        except KeyError as e:
            print('Key not found:', e)
            metadata['XScaleUnit'] = None
            metadata['YScaleUnit'] = None
        try:
            metadata['ZScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            # metadata['ZScale'] = np.round(metadata['ZScale'], 3)
            try:
                metadata['ZScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
            except KeyError as e:
                print('Key not found:', e)
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
        except Exception as e:
            # print('Exception:', e)
            if dim2none:
                metadata['ZScale'] = None
                metadata['ZScaleUnit'] = None
            if not dim2none:
                # set to isotropic scaling if it was single plane only
                metadata['ZScale'] = metadata['XScale']
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
    except Exception as e:
        print('Exception:', e)
        print('Scaling Data could not be found.')

    # try to get software version
    try:
        metadata['SW-Name'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Name']
        metadata['SW-Version'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Version']
    except KeyError as e:
        print('Key not found:', e)
        metadata['SW-Name'] = None
        metadata['SW-Version'] = None

    try:
        metadata['AcqDate'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['AcquisitionDateAndTime']
    except KeyError as e:
        print('Key not found:', e)
        metadata['AcqDate'] = None

    # get objective data
    try:
        metadata['ObjName'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Name']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjName'] = None

    try:
        metadata['ObjImmersion'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Immersion']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjImmersion'] = None

    try:
        metadata['ObjNA'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                     ['Instrument']['Objectives']['Objective']['LensNA'])
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjNA'] = None

    try:
        metadata['ObjID'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Id']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjID'] = None

    try:
        metadata['TubelensMag'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                           ['Instrument']['TubeLenses']['TubeLens']['Magnification'])
    except KeyError as e:
        print('Key not found:', e)
        metadata['TubelensMag'] = None

    try:
        metadata['ObjNominalMag'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                             ['Instrument']['Objectives']['Objective']['NominalMagnification'])
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjNominalMag'] = None

    try:
        if metadata['TubelensMag'] is not None:
            metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
        if metadata['TubelensMag'] is None:
            print('No TublensMag found. Use 1 instead')
            metadata['ObjMag'] = metadata['ObjNominalMag'] * 1.0

    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjMag'] = None

    # get detector information
    if isinstance(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'], list):
        num_detectors = len(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'])
    else:
        num_detectors = 1

    if num_detectors == 1:

        # check for detector ID
        try:
            metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                          ['Instrument']['Detectors']['Detector']['Id'])
        except KeyError as e:
            metadata['DetectorID'].append(None)

        # check for detector Name
        try:
            metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                            ['Instrument']['Detectors']['Detector']['Name'])
        except KeyError as e:
            metadata['DetectorName'].append(None)

        # check for detector model
        try:
            metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                             ['Instrument']['Detectors']['Detector']['Manufacturer']['Model'])
        except KeyError as e:
            metadata['DetectorModel'].append(None)

        # check for detector type
        try:
            metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                            ['Instrument']['Detectors']['Detector']['Type'])
        except KeyError as e:
            metadata['DetectorType'].append(None)

    if num_detectors > 1:
        for d in range(num_detectors):

            # check for detector ID
            try:
                metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                              ['Instrument']['Detectors']['Detector'][d]['Id'])
            except KeyError as e:
                metadata['DetectorID'].append(None)

            # check for detector Name
            try:
                metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector'][d]['Name'])
            except KeyError as e:
                metadata['DetectorName'].append(None)

            # check for detector model
            try:
                metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Detectors']['Detector'][d]['Manufacturer']['Model'])
            except KeyError as e:
                metadata['DetectorModel'].append(None)

            # check for detector type
            try:
                metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector'][d]['Type'])
            except KeyError as e:
                metadata['DetectorType'].append(None)

    # check for well information
    metadata['Well_ArrayNames'] = []
    metadata['Well_Indices'] = []
    metadata['Well_PositionNames'] = []
    metadata['Well_ColId'] = []
    metadata['Well_RowId'] = []
    metadata['WellCounter'] = None

    try:
        print('Trying to extract Scene and Well information if existing ...')
        # extract well information from the dictionary
        allscenes = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']

        # loop over all detected scenes
        for s in range(metadata['SizeS']):

            if metadata['SizeS'] == 1:
                well = allscenes
                try:
                    metadata['Well_ArrayNames'].append(allscenes['ArrayName'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    try:
                        metadata['Well_ArrayNames'].append(well['Name'])
                    except KeyError as e:
                        print('Key not found in Metadata Dictionary:', e, 'Using A1 instead')
                        metadata['Well_ArrayNames'].append('A1')

                try:
                    metadata['Well_Indices'].append(allscenes['Index'])
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_Indices'].append(1)

                try:
                    metadata['Well_PositionNames'].append(allscenes['Name'])
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_PositionNames'].append('P1')

                try:
                    metadata['Well_ColId'].append(np.int(allscenes['Shape']['ColumnIndex']))
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_ColId'].append(0)

                try:
                    metadata['Well_RowId'].append(np.int(allscenes['Shape']['RowIndex']))
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_RowId'].append(0)

                try:
                    # count the content of the list, e.g. how many time a certain well was detected
                    metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['WellCounter'].append(Counter({'A1': 1}))

            if metadata['SizeS'] > 1:
                try:
                    well = allscenes[s]
                    metadata['Well_ArrayNames'].append(well['ArrayName'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    try:
                        metadata['Well_ArrayNames'].append(well['Name'])
                    except KeyError as e:
                        print('Key not found in Metadata Dictionary:', e, 'Using A1 instead')
                        metadata['Well_ArrayNames'].append('A1')

                # get the well information
                try:
                    metadata['Well_Indices'].append(well['Index'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_Indices'].append(None)
                try:
                    metadata['Well_PositionNames'].append(well['Name'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_PositionNames'].append(None)

                try:
                    metadata['Well_ColId'].append(np.int(well['Shape']['ColumnIndex']))
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_ColId'].append(None)

                try:
                    metadata['Well_RowId'].append(np.int(well['Shape']['RowIndex']))
                except KeyError as e:
                    print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_RowId'].append(None)

                # count the content of the list, e.g. how many time a certain well was detected
                metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])

            # count the number of different wells
            metadata['NumWells'] = len(metadata['WellCounter'].keys())

    except (KeyError, TypeError) as e:
        print('No valid Scene or Well information found:', e)

    # close CZI file
    czi.close()

    # close AICSImage object
    czi_aics.close()

    return metadata


def get_additional_metadata_czi(filename):
    """
    Returns a dictionary with additional CZI metadata.

    :param filename: filename of the CZI image
    :type filename: str
    :return: additional_czimd - dictionary with additional CZI metainformation
    :rtype: dict
    """

    # get CZI object and read array
    czi = zis.CziFile(filename)

    # parse the XML into a dictionary
    metadatadict_czi = xmltodict.parse(czi.metadata())
    additional_czimd = {}

    try:
        additional_czimd['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['Experiment'] = None

    try:
        additional_czimd['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['HardwareSetting'] = None

    try:
        additional_czimd['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['CustomAttributes'] = None

    try:
        additional_czimd['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['DisplaySetting'] = None

    try:
        additional_czimd['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['Layers'] = None

    # close CZI file
    czi.close()

    return additional_czimd


def md2dataframe(metadata, paramcol='Parameter', keycol='Value'):
    """Convert the metadata dictionary to a Pandas DataFrame.

    :param metadata: MeteData dictionary
    :type metadata: dict
    :param paramcol: Name of Columns for the MetaData Parameters, defaults to 'Parameter'
    :type paramcol: str, optional
    :param keycol: Name of Columns for the MetaData Values, defaults to 'Value'
    :type keycol: str, optional
    :return: Pandas DataFrame containing all the metadata
    :rtype: Pandas.DataFrame
    """
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in metadata.keys():
        d = {'Parameter': k, 'Value': metadata[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def create_ipyviewer_ome_tiff(array, metadata):
    """
    Creates a simple interactive viewer inside a Jupyter Notebook.
    Works with OME-TIFF files and the respective metadata

    :param array: multidimensional array containing the pixel data
    :type array: NumPy.Array
    :param metadata: dictionary with the metainformation
    :return: out - interactive widgetsfor jupyter notebook
    :rtype: IPyWidgets Output
    :return: ui - ui for interactive widgets
    :rtype: IPyWidgets UI
    """

    # time slider
    t = widgets.IntSlider(description='Time:',
                          min=1,
                          max=metadata['SizeT'],
                          step=1,
                          value=1,
                          continuous_update=False)

    # zplane lsider
    z = widgets.IntSlider(description='Z-Plane:',
                          min=1,
                          max=metadata['SizeZ'],
                          step=1,
                          value=1,
                          continuous_update=False)

    # channel slider
    c = widgets.IntSlider(description='Channel:',
                          min=1,
                          max=metadata['SizeC'],
                          step=1,
                          value=1)

    # slider for contrast
    r = widgets.IntRangeSlider(description='Display Range:',
                               min=array.min(),
                               max=array.max(),
                               step=1,
                               value=[array.min(), array.max()],
                               continuous_update=False)

    # disable slider that are not needed
    if metadata['SizeT'] == 1:
        t.disabled = True
    if metadata['SizeZ'] == 1:
        z.disabled = True
    if metadata['SizeC'] == 1:
        c.disabled = True

    sliders = metadata['DimOrder BF Array'][:-2] + 'R'

    # TODO: this section is not complete, because it does not contain all possible cases
    # TODO: it is still under constrcution and can be done probably in a much smarter way

    if sliders == 'CTZR':
        ui = widgets.VBox([c, t, z, r])

        def get_TZC_czi(c_ind, t_ind, z_ind, r):
            display_image(array, metadata, sliders, c=c_ind, t=t_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 't_ind': t, 'z_ind': z, 'r': r})

    if sliders == 'TZCR':
        ui = widgets.VBox([t, z, c, r])

        def get_TZC_czi(t_ind, z_ind, c_ind, r):
            display_image(array, metadata, sliders, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'TCZR':
        ui = widgets.VBox([t, c, z, r])

        def get_TZC_czi(t_ind, c_ind, z_ind, r):
            display_image(array, metadata, sliders, t=t_ind, c=t_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'CZTR':
        ui = widgets.VBox([c, z, t, r])

        def get_TZC_czi(c_ind, z_ind, t_ind, r):
            display_image(array, metadata, sliders, c=c_ind, z=z_ind, t=t_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 'z_ind': z, 't_ind': t, 'r': r})

    if sliders == 'ZTCR':
        ui = widgets.VBox([z, t, c, r])

        def get_TZC_czi(z_ind, t_ind, c_ind, r):
            display_image(array, metadata, sliders, z=z_ind, t=t_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'z_ind': z, 't_ind': t, 'c_ind': c, 'r': r})

    if sliders == 'ZCTR':
        ui = widgets.VBox([z, c, t, r])

        def get_TZC_czi(z_ind, c_ind, t_ind, r):
            display_image(array, metadata, sliders, z=z_ind, c=c_ind, t=t_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'z_ind': z, 'c_ind': c, 't_ind': t, 'r': r})

    """
    ui = widgets.VBox([t, z, c, r])

    def get_TZC_ometiff(t, z, c, r):
        display_image(array, metadata, 'TZCR', t=t, z=z, c=c, vmin=r[0], vmax=r[1])

    out = widgets.interactive_output(get_TZC_ometiff, {'t': t, 'z': z, 'c': c, 'r': r})
    """

    return out, ui  # , t, z, c, r


def create_ipyviewer_czi(cziarray, metadata):
    """
    Creates a simple interactive viewer inside a Jupyter Notebook.
    Works with CZI files and the respective metadata

    :param array: multidimensional array containing the pixel data
    :type array: NumPy.Array
    :param metadata: dictionary with the metainformation
    :return: out - interactive widgetsfor jupyter notebook
    :rtype: IPyWidgets Output
    :return: ui - ui for interactive widgets
    :rtype: IPyWidgets UI
    """

    dim_dict = metadata['DimOrder CZI']

    useB = False
    useS = False

    if 'B' in dim_dict and dim_dict['B'] >= 0:
        useB = True
        b = widgets.IntSlider(description='Blocks:',
                              min=1,
                              max=metadata['SizeB'],
                              step=1,
                              value=1,
                              continuous_update=False)

    if 'S' in dim_dict and dim_dict['S'] >= 0:
        useS = True
        s = widgets.IntSlider(description='Scenes:',
                              min=1,
                              max=metadata['SizeS'],
                              step=1,
                              value=1,
                              continuous_update=False)

    t = widgets.IntSlider(description='Time:',
                          min=1,
                          max=metadata['SizeT'],
                          step=1,
                          value=1,
                          continuous_update=False)

    z = widgets.IntSlider(description='Z-Plane:',
                          min=1,
                          max=metadata['SizeZ'],
                          step=1,
                          value=1,
                          continuous_update=False)

    c = widgets.IntSlider(description='Channel:',
                          min=1,
                          max=metadata['SizeC'],
                          step=1,
                          value=1)

    print(cziarray.min(), cziarray.max())

    r = widgets.IntRangeSlider(description='Display Range:',
                               min=cziarray.min(),
                               max=cziarray.max(),
                               step=1,
                               value=[cziarray.min(), cziarray.max()],
                               continuous_update=False)

    # disable slider that are not needed
    if metadata['SizeB'] == 1 and useB:
        b.disabled = True
    if metadata['SizeS'] == 1 and useS:
        s.disabled = True
    if metadata['SizeT'] == 1:
        t.disabled = True
    if metadata['SizeZ'] == 1:
        z.disabled = True
    if metadata['SizeC'] == 1:
        c.disabled = True

    sliders = metadata['Axes'][:-3] + 'R'

    # TODO: this section is not complete, because it does not contain all possible cases
    # TODO: it is still under constrcution and can be done probably in a much smarter way

    if sliders == 'BTZCR':
        ui = widgets.VBox([b, t, z, c, r])

        def get_TZC_czi(b_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'BTCZR':
        ui = widgets.VBox([b, t, c, z, r])

        def get_TZC_czi(b_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'BSTZCR':
        ui = widgets.VBox([b, s, t, z, c, r])

        def get_TZC_czi(b_ind, s_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'BSCR':
        ui = widgets.VBox([b, s, c, r])

        def get_TZC_czi(b_ind, s_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 'c_ind': c, 'r': r})

    if sliders == 'BSTCZR':
        ui = widgets.VBox([b, s, t, c, z, r])

        def get_TZC_czi(b_ind, s_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'STZCR':
        ui = widgets.VBox([s, t, z, c, r])

        def get_TZC_czi(s_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'STCZR':
        ui = widgets.VBox([s, t, c, z, r])

        def get_TZC_czi(s_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'TZCR':
        ui = widgets.VBox([t, z, c, r])

        def get_TZC_czi(t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'TCZR':
        ui = widgets.VBox([t, c, z, r])

        def get_TZC_czi(t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'SCR':
        ui = widgets.VBox([s, c, r])

        def get_TZC_czi(s_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 'c_ind': c, 'r': r})

    if sliders == 'ZR':
        ui = widgets.VBox([z, r])

        def get_TZC_czi(z_ind, r):
            display_image(cziarray, metadata, sliders, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'z_ind': z, 'r': r})

    if sliders == 'TR':
        ui = widgets.VBox([t, r])

        def get_TZC_czi(t_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'r': r})

    if sliders == 'CR':
        ui = widgets.VBox([c, r])

        def get_TZC_czi(c_ind, r):
            display_image(cziarray, metadata, sliders, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 'r': r})

    if sliders == 'BTCR':
        ui = widgets.VBox([b, t, c, r])

        def get_TZC_czi(b_ind, t_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'c_ind': c, 'r': r})

    ############### Lightsheet data #################

    if sliders == 'VIHRSCTZR':
        ui = widgets.VBox([c, t, z, r])

        def get_TZC_czi(c_ind, t_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, c=c_ind, t=t_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 't_ind': t, 'z_ind': z, 'r': r})

    return out, ui


def display_image(array, metadata, sliders,
                  b=0,
                  s=0,
                  m=0,
                  t=0,
                  c=0,
                  z=0,
                  vmin=0,
                  vmax=1000):
    """Displays the CZI or OME-TIFF image using a simple interactive viewer
    inside a Jupyter Notebook with dimension sliders.

    :param array:  multidimensional array containing the pixel data
    :type array: NumPy.Array
    :param metadata: dictionary with the metainformation
    :type metadata: dict
    :param sliders: string specifying the required sliders
    :type sliders: str
    :param b: block index of plan to be displayed, defaults to 0
    :type b: int, optional
    :param s: scene index of plan to be displayed, defaults to 0
    :type s: int, optional
    :param m: tile index of plan to be displayed, defaults to 0
    :type m: int, optional
    :param t: time index of plan to be displayed, defaults to 0
    :type t: int, optional
    :param c: channel index of plan to be displayed, defaults to 0
    :type c: int, optional
    :param z: zplane index of plan to be displayed, defaults to 0
    :type z: int, optional
    :param vmin: minimum value for scaling, defaults to 0
    :type vmin: int, optional
    :param vmax: maximum value for scaling, defaults to 1000
    :type vmax: int, optional
    """

    dim_dict = metadata['DimOrder CZI']

    if metadata['ImageType'] == 'ometiff':

        if sliders == 'TZCR':
            image = array[t - 1, z - 1, c - 1, :, :]

        if sliders == 'CTZR':
            image = array[c - 1, t - 1, z - 1, :, :]

        if sliders == 'TCZR':
            image = array[t - 1, c - 1, z - 1, :, :]

        if sliders == 'CZTR':
            image = array[c - 1, z - 1, t - 1, :, :]

        if sliders == 'ZTCR':
            image = array[z - 1, t - 1, c - 1, :, :]

        if sliders == 'ZCTR':
            image = array[z - 1, c - 1, z - 1, :, :]

    if metadata['ImageType'] == 'czi':

        # add more dimension orders when needed
        if sliders == 'BTZCR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'BTCZR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'BSTZCR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'BSTCZR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'STZCR':
            if metadata['isRGB']:
                image = array[s - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[s - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'STCZR':
            if metadata['isRGB']:
                image = array[s - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[s - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'TZCR':
            if metadata['isRGB']:
                image = array[t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[t - 1, z - 1, c - 1, :, :]

        if sliders == 'TCZR':
            if metadata['isRGB']:
                image = array[t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[t - 1, c - 1, z - 1, :, :]

        if sliders == 'SCR':
            if metadata['isRGB']:
                image = array[s - 1, c - 1, :, :, :]
            else:
                image = array[s - 1, c - 1, :, :]

        if sliders == 'ZR':
            if metadata['isRGB']:
                image = array[z - 1, :, :, :]
            else:
                image = array[z - 1, :, :]

        if sliders == 'TR':
            if metadata['isRGB']:
                image = array[t - 1, :, :, :]
            else:
                image = array[t - 1, :, :]

        if sliders == 'CR':
            if metadata['isRGB']:
                image = array[c - 1, :, :, :]
            else:
                image = array[c - 1, :, :]

        if sliders == 'BSCR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, c - 1, :, :]

        if sliders == 'BTCR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, c - 1, :, :]

        ####### lightsheet Data #############
        if sliders == 'VIHRSCTZR':
            # reduce dimensions
            image = np.squeeze(array, axis=(0, 1, 2, 3, 4))
            image = image[c - 1, t - 1, z - 1, :, :]

    # display the labeled image
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cm.gray)
    fig.colorbar(im, cax=cax, orientation='vertical')
    print('Min-Max (Current Plane):', image.min(), '-', image.max())


def get_dimorder(dimstring):
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
    dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', 'T', 'C', 'Z', 'Y', 'X', '0']
    dims_dict = {}

    # loop over all dimensions and find the index
    for d in dims:

        dims_dict[d] = dimstring.find(d)
        dimindex_list.append(dimstring.find(d))

    # check if a dimension really exists
    numvalid_dims = sum(i > 0 for i in dimindex_list)

    return dims_dict, dimindex_list, numvalid_dims


def get_array_czi(filename,
                  replace_value=False,
                  remove_HDim=True,
                  return_addmd=False):
    """Get the pixel data of the CZI file as multidimensional NumPy.Array

    :param filename: filename of the CZI file
    :type filename: str
    :param replacevalue: replace arrays entries with a specific value with NaN, defaults to False
    :type replacevalue: bool, optional
    :param remove_HDim: remove the H-Dimension (Airy Scan Detectors), defaults to True
    :type remove_HDim: bool, optional
    :param return_addmd: read the additional metadata, defaults to False
    :type return_addmd: bool, optional
    :return: cziarray - dictionary with the dimensions and its positions
    :rtype: NumPy.Array
    :return: metadata - dictionary with CZI metadata
    :rtype: dict
    :return: additional_metadata_czi - dictionary with additional CZI metadata
    :rtype: dict
    """

    metadata = get_metadata_czi(filename)
    additional_metadata_czi = get_additional_metadata_czi(filename)

    # get CZI object and read array
    czi = zis.CziFile(filename)
    cziarray = czi.asarray()

    # check for H dimension and remove
    if remove_HDim and metadata['Axes_czifile'][0] == 'H':
        # metadata['Axes'] = metadata['Axes_czifile'][1:]
        metadata['Axes_czifile'] = metadata['Axes_czifile'].replace('H', '')
        cziarray = np.squeeze(cziarray, axis=0)

    # get additional information about dimension order etc.
    dim_dict, dim_list, numvalid_dims = get_dimorder(metadata['Axes_czifile'])
    metadata['DimOrder CZI'] = dim_dict

    if cziarray.shape[-1] == 3:
        pass
    else:
        # remove the last dimension from the end
        cziarray = np.squeeze(cziarray, axis=len(metadata['Axes_czifile']) - 1)
        metadata['Axes_czifile'] = metadata['Axes_czifile'].replace('0', '')

    if replace_value:
        cziarray = replace_value(cziarray, value=0)

    # close czi file
    czi.close()

    return cziarray, metadata, additional_metadata_czi


def replace_value(data, value=0):
    """Replace specifc values in array with NaN

    :param data: Array where values should be replaced
    :type data: NumPy.Array
    :param value: value inside array to be replaced with NaN, defaults to 0
    :type value: int, optional
    :return: array with new values
    :rtype: NumPy.Array
    """

    data = data.astype('float')
    data[data == value] = np.nan

    return data


def get_scalefactor(metadata):
    """Add scaling factors to the metadata dictionary

    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :return: dictionary with additional keys for scling factors
    :rtype: dict
    """

    # set default scale factor to 1.0
    scalefactors = {'xy': 1.0,
                    'zx': 1.0
                    }

    try:
        # get the factor between XY scaling
        scalefactors['xy'] = metadata['XScale'] / metadata['YScale']
        # get the scalefactor between XZ scaling
        scalefactors['zx'] = metadata['ZScale'] / metadata['YScale']
    except KeyError as e:
        print('Key not found: ', e, 'Using defaults = 1.0')

    return scalefactors


def show_napari(array, metadata,
                blending='additive',
                gamma=0.85,
                verbose=True,
                use_pylibczi=True,
                rename_sliders=False):
    """Show the multidimensional array using the Napari viewer

    :param array: multidimensional NumPy.Array containing the pixeldata
    :type array: NumPy.Array
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :param blending: NapariViewer option for blending, defaults to 'additive'
    :type blending: str, optional
    :param gamma: NapariViewer value for Gamma, defaults to 0.85
    :type gamma: float, optional
    :param verbose: show additional output, defaults to True
    :type verbose: bool, optional
    :param use_pylibczi: specify if pylibczi was used to read the CZI file, defaults to True
    :type use_pylibczi: bool, optional
    :param rename_sliders: name slider with correct labels output, defaults to False
    :type verbose: bool, optional
    """

    def calc_scaling(data, corr_min=1.0,
                     offset_min=0,
                     corr_max=0.85,
                     offset_max=0):

        # get min-max values for initial scaling
        minvalue = np.round((data.min() + offset_min) * corr_min)
        maxvalue = np.round((data.max() + offset_max) * corr_max)
        print('Scaling: ', minvalue, maxvalue)

    with napari.gui_qt():

        # create scalefcator with all ones
        scalefactors = [1.0] * len(array.shape)

        # initialize the napari viewer
        print('Initializing Napari Viewer ...')
        viewer = napari.Viewer()

        if metadata['ImageType'] == 'ometiff':

            # find position of dimensions
            posZ = metadata['DimOrder BF Array'].find('Z')
            posC = metadata['DimOrder BF Array'].find('C')
            posT = metadata['DimOrder BF Array'].find('T')

            # get the scalefactors from the metadata
            scalef = get_scalefactor(metadata)
            # modify the tuple for the scales for napari
            scalefactors[posZ] = scalef['zx']

            if verbose:
                print('Dim PosT : ', posT)
                print('Dim PosC : ', posC)
                print('Dim PosZ : ', posZ)
                print('Scale Factors : ', scalefactors)

            # add all channels as layers
            for ch in range(metadata['SizeC']):

                try:
                    # get the channel name
                    chname = metadata['Channels'][ch]
                except:
                    # or use CH1 etc. as string for the name
                    chname = 'CH' + str(ch + 1)

                # cutout channel
                channel = array.take(ch, axis=posC)
                print('Shape Channel : ', ch, channel.shape)

                # actually show the image array
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                clim = [channel.min(), np.round(channel.max() * 0.85)]
                if verbose:
                    print('Scaling: ', clim)
                viewer.add_image(channel,
                                 name=chname,
                                 scale=scalefactors,
                                 contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma)

        if metadata['ImageType'] == 'czi':

            if not use_pylibczi:
                # use find position of dimensions
                # posZ = metadata['Axes'].find('Z')
                # posC = metadata['Axes'].find('C')
                # posT = metadata['Axes'].find('T')
                dimpos = get_dimpositions(metadata['Axes'])

            if use_pylibczi:
                # posZ = metadata['Axes_aics'].find('Z')
                # posC = metadata['Axes_aics'].find('C')
                # posT = metadata['Axes_aics'].find('T')
                dimpos = get_dimpositions(metadata['Axes_aics'])

            # get the scalefactors from the metadata
            scalef = get_scalefactor(metadata)
            # modify the tuple for the scales for napari
            # temporary workaround for slider / floating point issue
            # https://forum.image.sc/t/problem-with-dimension-slider-when-adding-array-as-new-layer-for-ome-tiff/39092/2?u=sebi06
            scalef['zx'] = np.round(scalef['zx'], 3)

            # modify the tuple for the scales for napari
            scalefactors[dimpos['Z']] = scalef['zx']

            # remove C dimension from scalefactor
            scalefactors_ch = scalefactors.copy()
            del scalefactors_ch[dimpos['C']]

            if metadata['SizeC'] > 1:
                # add all channels as layers
                for ch in range(metadata['SizeC']):

                    try:
                        # get the channel name
                        chname = metadata['Channels'][ch]
                    except:
                        # or use CH1 etc. as string for the name
                        chname = 'CH' + str(ch + 1)

                    # cut out channel
                    # use dask if array is a dask.array
                    if isinstance(array, da.Array):
                        print('Extract Channel using Dask.Array')
                        channel = array.compute().take(ch, axis=dimpos['C'])
                        new_dimstring = metadata['Axes_aics'].replace('C', '')

                    else:
                        # use normal numpy if not
                        print('Extract Channel NumPy.Array')
                        channel = array.take(ch, axis=dimpos['C'])
                        if not use_pylibczi:
                            new_dimstring = metadata['Axes'].replace('C', '')
                        if use_pylibczi:
                            new_dimstring = metadata['Axes_aics'].replace('C', '')

                    # actually show the image array
                    print('Adding Channel  : ', chname)
                    print('Shape Channel   : ', ch, channel.shape)
                    print('Scaling Factors : ', scalefactors_ch)

                    # get min-max values for initial scaling
                    clim = calc_scaling(channel,
                                        corr_min=1.0,
                                        offset_min=0,
                                        corr_max=0.85,
                                        offset_max=0)

                    viewer.add_image(channel,
                                     name=chname,
                                     scale=scalefactors,
                                     contrast_limits=clim,
                                     blending=blending,
                                     gamma=gamma)

            if metadata['SizeC'] == 1:

                # just add one channel as a layer
                try:
                    # get the channel name
                    chname = metadata['Channels'][0]
                except:
                    # or use CH1 etc. as string for the name
                    chname = 'CH' + str(ch + 1)

                # actually show the image array
                print('Adding Channel: ', chname)
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                # clim = calc_scaling(array)

                viewer.add_image(array,
                                 name=chname,
                                 scale=scalefactors,
                                 # contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma,
                                 is_pyramid=False)

        if rename_sliders:

            print('Renaming the Sliders based on the Dimension String ....')

            # get the position of dimension entries after removing C dimension
            dimpos_viewer = get_dimpositions(new_dimstring)

            # get the label of the sliders
            sliders = viewer.dims.axis_labels

            # update the labels with the correct dimension strings
            slidernames = ['B', 'S', 'T', 'Z']
            for s in slidernames:
                if dimpos_viewer[s] >= 0:
                    sliders[dimpos_viewer[s]] = s
            # apply the new labels to the viewer
            viewer.dims.axis_labels = sliders


def check_for_previewimage(czi):
    """Check if the CZI contains an image from a prescan camera

    :param czi: CZI imagefile object
    :type metadata: CziFile object
    :return: has_attimage - Boolean if CZI image contains prescan image
    :rtype: bool
    """

    att = []

    # loop over the attachments
    for attachment in czi.attachments():
        entry = attachment.attachment_entry
        print(entry.name)
        att.append(entry.name)

    has_attimage = False

    # check for the entry "SlidePreview"
    if 'SlidePreview' in att:
        has_attimage = True

    return has_attimage


def writexml_czi(filename, xmlsuffix='_CZI_MetaData.xml'):
    """Write XML imformation of CZI to disk

    :param filename: CZI image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    # open czi file and get the metadata
    czi = zis.CziFile(filename)
    mdczi = czi.metadata()
    czi.close()

    # change file name
    xmlfile = filename.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(mdczi))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile


def writexml_ometiff(filename, xmlsuffix='_OMETIFF_MetaData.xml'):
    """Write XML imformation of OME-TIFF to disk

    :param filename: OME-TIFF image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_OMETIFF_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    if filename.lower().endswith('.ome.tiff'):
        ext = '.ome.tiff'
    if filename.lower().endswith('.ome.tif'):
        ext = '.ome.tif'

    with tifffile.TiffFile(filename) as tif:
        # omexml_string = tif[0].image_description.decode('utf-8')
        omexml_string = tif[0].image_description

    # get tree from string
    # tree = ET.ElementTree(ET.fromstring(omexml_string.encode('utf-8')))
    tree = ET.ElementTree(ET.fromstring(omexml_string))

    # change file name
    xmlfile = filename.replace(ext, xmlsuffix)

    tree.write(xmlfile, encoding='utf-8', method='xml', pretty_print=True)
    print('Created OME-XML file for testdata: ', filename)

    return xmlfile


def getImageSeriesIDforWell(welllist, wellID):
    """
    Returns all ImageSeries (for OME-TIFF) indicies for a specific wellID

    :param welllist: list containing all wellIDs as stringe, e.g. '[B4, B4, B4, B4, B5, B5, B5, B5]'
    :type welllist: list
    :param wellID: string specifying the well, eg.g. 'B4'
    :type wellID: str
    :return: imageseriesindices - list containing all ImageSeries indices, which correspond the the well
    :rtype: list
    """

    imageseries_indices = [i for i, x in enumerate(welllist) if x == wellID]

    return imageseries_indices


def addzeros(number):
    """Convert a number into a string and add leading zeros.
    Typically used to construct filenames with equal lengths.

    :param number: the number
    :type number: int
    :return: zerostring - string with leading zeros
    :rtype: str
    """

    if number < 10:
        zerostring = '0000' + str(number)
    if number >= 10 and number < 100:
        zerostring = '000' + str(number)
    if number >= 100 and number < 1000:
        zerostring = '00' + str(number)
    if number >= 1000 and number < 10000:
        zerostring = '0' + str(number)

    return zerostring


def write_ometiff(filepath, img,
                  scalex=0.1,
                  scaley=0.1,
                  scalez=1.0,
                  dimorder='TZCYX',
                  pixeltype=np.uint16,
                  swapxyaxes=True,
                  series=1):
    """ONLY FOR INTERNAL TESTING - DO NOT USE!

    This function will write an OME-TIFF file to disk.
    The out 6D array has the following dimension order:

    [T, Z, C, Y, X] if swapxyaxes = True

    [T, Z, C, X, Y] if swapxyaxes = False
    """

    # Dimension STZCXY
    if swapxyaxes:
        # swap xy to write the OME-Stack with the correct shape
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[4]
        SizeY = img.shape[3]

    if not swapxyaxes:
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[3]
        SizeY = img.shape[4]

    # Getting metadata info
    omexml = bioformats.omexml.OMEXML()
    omexml.image(series - 1).Name = filepath

    for s in range(series):
        p = omexml.image(s).Pixels
        p.ID = str(s)
        p.SizeX = SizeX
        p.SizeY = SizeY
        p.SizeC = SizeC
        p.SizeT = SizeT
        p.SizeZ = SizeZ
        p.PhysicalSizeX = np.float(scalex)
        p.PhysicalSizeY = np.float(scaley)
        p.PhysicalSizeZ = np.float(scalez)
        if pixeltype == np.uint8:
            p.PixelType = 'uint8'
        if pixeltype == np.uint16:
            p.PixelType = 'uint16'
        p.channel_count = SizeC
        p.plane_count = SizeZ * SizeT * SizeC
        p = writeOMETIFFplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder)

        for c in range(SizeC):
            # if pixeltype == 'unit8':
            if pixeltype == np.uint8:
                p.Channel(c).SamplesPerPixel = 1

            if pixeltype == np.uint16:
                p.Channel(c).SamplesPerPixel = 2

        omexml.structured_annotations.add_original_metadata(bioformats.omexml.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml(encoding='utf-8')

    # write file and save OME-XML as description
    tifffile.imwrite(filepath, img, metadata={'axes': dimorder}, description=xml)

    return filepath


def writeOMETIFFplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCXY', verbose=False):
    """ONLY FOR INTERNAL TESTING - DO NOT USE!

    """
    if order == 'TZCYX' or order == 'TZCXY':

        pixel.DimensionOrder = bioformats.omexml.DO_XYCZT
        counter = 0
        for t in range(SizeT):
            for z in range(SizeZ):
                for c in range(SizeC):

                    if verbose:
                        print('Write PlaneTable: ', t, z, c),
                        sys.stdout.flush()

                    pixel.Plane(counter).TheT = t
                    pixel.Plane(counter).TheZ = z
                    pixel.Plane(counter).TheC = c
                    counter = counter + 1

    return pixel


def write_ometiff_aicsimageio(savepath, imgarray, metadata,
                              reader='aicsimageio',
                              overwrite=False):
    """Write an OME-TIFF file from an image array based on the metadata.

    :param filepath: savepath of the OME-TIFF stack
    :type filepath: str
    :param imgarray: multi-dimensional image array
    :type imgarray: NumPy.Array
    :param metadata: metadata dictionary with the required information
    to create an correct OME-TIFF file
    :type metadata: dict
    :param reader: string (aicsimagio or czifile) specifying
    the used reader, defaults to aicsimageio
    :type metadata: str
    :param overwrite: option to overwrite an existing OME-TIFF, defaults to False
    :type overwrite: bool, optional
    """

    # define scaling from metadata or use defualt scaling
    try:
        pixels_physical_size = [metadata['XScale'],
                                metadata['YScale'],
                                metadata['ZScale']]
    except KeyError as e:
        print('Key not found:', e)
        print('Use default scaling XYZ=1.0')
        pixels_physical_size = [1.0, 1.0, 1.0]

    # define channel names list from metadata
    try:
        channel_names = []
        for ch in metadata['Channels']:
            channel_names.append(ch)
    except KeyError as e:
        print('Key not found:', e)
        channel_names = None

    # get the dimensions and their position inside the dimension string
    if reader == 'aicsimageio':

        dims_dict, dimindex_list, numvalid_dims = get_dimorder(metadata['Axes_aics'])

        # if the array has more than 5 dimensions then remove the S dimension
        # because it is not supported by OME-TIFF
        if len(imgarray.shape) > 5:
            try:
                imgarray = np.squeeze(imgarray, axis=dims_dict['S'])
            except Exception:
                print('Could not remover S Dimension from string.)')

        # remove the S character from the dimension string
        new_dimorder = metadata['Axes_aics'].replace('S', '')

    if reader == 'czifile':

        new_dimorder = metadata['Axes']
        dims_dict, dimindex_list, numvalid_dims = get_dimorder(metadata['Axes'])
        """
        '0': 'Sample',  # e.g. RGBA
        'X': 'Width',
        'Y': 'Height',
        'C': 'Channel',
        'Z': 'Slice',  # depth
        'T': 'Time',
        'R': 'Rotation',
        'S': 'Scene',  # contiguous regions of interest in a mosaic image
        'I': 'Illumination',  # direction
        'B': 'Block',  # acquisition
        'M': 'Mosaic',  # index of tile for compositing a scene
        'H': 'Phase',  # e.g. Airy detector fibers
        'V': 'View',  # e.g. for SPIM
        """

        to_remove = []

        # list of unspupported dims for writing an OME-TIFF
        dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', '0']

        for dim in dims:
            if dims_dict[dim] >= 0:
                # remove the CZI DIMENSION character from the dimension string
                new_dimorder = new_dimorder.replace(dim, '')
                # add dimension index to the list of axis to be removed
                to_remove.append(dims_dict[dim])
                print('Remove Dimension:', dim)

        # create tuple with dimensions to be removed
        dims2remove = tuple(to_remove)
        # remove dimensions from array
        imgarray = np.squeeze(imgarray, axis=dims2remove)

    # write the array as an OME-TIFF incl. the metadata
    try:
        with ome_tiff_writer.OmeTiffWriter(savepath, overwrite_file=overwrite) as writer:
            writer.save(imgarray,
                        channel_names=channel_names,
                        image_name=os.path.basename((savepath)),
                        pixels_physical_size=pixels_physical_size,
                        channel_colors=None,
                        dimension_order=new_dimorder)
            writer.close()
    except Exception as error:
        print(error.__class__.__name__ + ": " + error.msg)
        print('Could not write OME-TIFF')
        savepath = None

    return savepath


def correct_omeheader(omefile,
                      old=("2012-03", "2013-06", r"ome/2016-06"),
                      new=("2016-06", "2016-06", r"OME/2016-06")
                      ):
    """This function is actually a workaround for AICSImageIO<=3.1.4 that
    correct some incorrect namespaces inside the OME-XML header

    :param omefile: OME-TIFF image file
    :type omefile: string
    :param old: strings that should be corrected, defaults to ("2012-03", "2013-06", r"ome/2016-06")
    :type old: tuple, optional
    :param new: replacement for the strings to be corrected, defaults to ("2016-06", "2016-06", r"OME/2016-06")
    :type new: tuple, optional
    """

    # create the tif object from the filename
    tif = tifffile.TiffFile(omefile)

    # get the pixel array and the OME-XML string
    array = tif.asarray()
    omexml_string = tif.ome_metadata

    # search for the strings to be replaced and do it
    for ostr, nstr in zip(old, new):
        print('Replace: ', ostr, 'with', nstr)
        omexml_string = omexml_string.replace(ostr, nstr)

    # save the file with the new, correct strings
    tifffile.imsave(omefile, array,
                    photometric='minisblack',
                    description=omexml_string)

    # close tif object
    tif.close()

    print('Updated OME Header.')


def get_fname_woext(filepath):
    """Get the complete path of a file without the extension
    It alos will works for extensions like c:\myfile.abc.xyz
    The output will be: c:\myfile

    :param filepath: complete fiepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
    """
    # create empty string
    real_extension = ''

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remover real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def convert_to_ometiff(imagefilepath,
                       bftoolsdir='/Users/bftools',
                       czi_include_attachments=False,
                       czi_autostitch=True,
                       verbose=True):
    """Convert image file using bfconvert tool into a OME-TIFF from with a python script.

    :param imagefilepath: path to imagefile
    :type imagefilepath: str
    :param bftoolsdir: bftools directory containing the bfconvert, defaults to '/Users/bftools'
    :type bftoolsdir: str, optional
    :param czi_include_attachments: option convert a CZI attachment (if CZI), defaults to False
    :type czi_include_attachments: bool, optional
    :param czi_autostitch: option stich a CZI, defaults to True
    :type czi_autostitch: bool, optional
    :param verbose: show additional output, defaults to True
    :type verbose: bool, optional
    :return: fileparh of created OME-TIFF file
    :rtype: str
    """
    # check if path exits
    if not os.path.exists(bftoolsdir):
        print('No bftools dirctory found. Nothing will be converted')
        file_ometiff = None

    if os.path.exists(bftoolsdir):

        # set working dir
        os.chdir(bftoolsdir)

        # get the imagefile path without extension
        imagefilepath_woext = get_fname_woext(imagefilepath)

        # create imagefile path for OME-TIFF
        file_ometiff = imagefilepath_woext + '.ome.tiff'

        # create cmdstring for CZI files- mind the spaces !!!
        if imagefilepath.lower().endswith('.czi'):

            # configure the CZI options
            if czi_include_attachments:
                czi_att = 'true'
            if not czi_include_attachments:
                czi_att = 'false'

            if czi_autostitch:
                czi_stitch = 'true'
            if not czi_autostitch:
                czi_stitch = 'false'

            # create cmdstring - mind the spaces !!!
            cmdstring = 'bfconvert -no-upgrade -option zeissczi.attachments ' + czi_att + ' -option zeissczi.autostitch ' + \
                czi_stitch + ' "' + imagefilepath + '" "' + file_ometiff + '"'

        else:
            # create cmdstring for non-CZIs- mind the spaces !!!
            cmdstring = 'bfconvert -no-upgrade' + ' "' + imagefilepath + '" "' + file_ometiff + '"'

        if verbose:
            print('Original ImageFile : ', imagefilepath_woext)
            print('ImageFile OME.TIFF : ', file_ometiff)
            print('Use CMD : ', cmdstring)

        # run the bfconvert tool with the specified parameters
        os.system(cmdstring)
        print('Done.')

    return file_ometiff


def get_dimpositions(dimstring, tocheck=['B', 'S', 'T', 'Z', 'C']):
    """Simple function to get the indices of the dimension identifiers in a string

    :param dimstring: dimension string
    :type dimstring: str
    :param tocheck: list of entries to check, defaults to ['B', 'S', 'T', 'Z', 'C']
    :type tocheck: list, optional
    :return: dictionary with positions of dimensions inside string
    :rtype: dict
    """
    dimpos = {}
    for p in tocheck:
        dimpos[p] = dimstring.find(p)

    return dimpos
