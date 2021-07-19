from pylibCZIrw import czi as pyczi
from aicspylibczi import CziFile
import xmltodict
import pydash
import os
import sys
import itertools as it
import numpy as np
from collections import Counter
import pandas as pd



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
                'SizeX': 1,
                'SizeY': 1,
                'SizeZ': None,
                'SizeC': None,
                'SizeT': None,
                'SizeS': None,
                'SizeB': None,
                'SizeM': None,
                'isRGB': False,
                'isMosaic': False,
                'czi_size': None,
                'czi_dims': None,
                'czi_dims_shape': None,
                'ObjNA': [],
                'ObjMag': [],
                'ObjID': [],
                'ObjName': [],
                'ObjImmersion': [],
                'TubelensMag': [],
                'ObjNominalMag': [],
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
                'bbox_all_scenes': None,
                'bbox_all_mosaic_scenes': None,
                'bbox_all_mosaic_tiles': None,
                'bbox_all_tiles': None
                }

    return metadata


def get_metadata_czi(filename, dim2none=False, convert_scunit=True):
    """
    Returns a dictionary with CZI metadata.

    Information CZI Dimension Characters:
    - '0':'Sample',  # e.g. RGBA
    - 'X':'Width',
    - 'Y':'Height',
    - 'C':'Channel',
    - 'Z':'Slice',  # depth
    - 'T':'Time',
    - 'R':'Rotation',
    - 'S':'Scene',  # contiguous regions of interest in a mosaic image
    - 'I':'Illumination',  # direction
    - 'B':'Block',  # acquisition
    - 'M':'Mosaic',  # index of tile for compositing a scene
    - 'H':'Phase',  # e.g. Airy detector fibers
    - 'V':'View',  # e.g. for SPIM

    :param filename: filename of the CZI image
    :type filename: str
    :param dim2none: option to set non-existing dimension to None, defaults to False
    :type dim2none: bool, optional
    :param convert_scunit: convert scale unit string from 'µm' to 'micron', defaults to False
    :type convert_scunit: bool, optional
    :return: metadata, metadata_add - dictionaries with the relevant CZI metainformation
    :rtype: dict
    """

    # get metadata dictionary using pylibCZIrw
    czidoc = pyczi.open_czi(filename)
    metadatadict_czi = xmltodict.parse(czidoc.raw_metadata)

    # get czi object aicspylibczi
    aicsczi = CziFile(filename)

    # initialize metadata dictionary
    metadata = create_metadata_dict()

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['Extension'] = 'czi'
    metadata['ImageType'] = 'czi'

    # get additional data by using aicspylibczi directly
    metadata['aicsczi_dims'] = aicsczi.dims
    metadata['aicsczi_dims_shape'] = aicsczi.get_dims_shape()
    metadata['aicsczi_size'] = aicsczi.size
    metadata['isMosaic'] = aicsczi.is_mosaic()
    print('CZI is Mosaic :', metadata['isMosaic'])

    # get the dimensions of the bounding boxes for the scenes
    # metadata['BBoxes_Scenes'] = getbboxes_allscenes(czi, metadata, numscenes=metadata['SizeS'])

    metadata['bbox_all_scenes'] = aicsczi.get_all_scene_bounding_boxes()
    if aicsczi.is_mosaic():
        metadata['bbox_all_mosaic_scenes'] = aicsczi.get_all_mosaic_scene_bounding_boxes()
        metadata['bbox_all_mosaic_tiles'] = aicsczi.get_all_mosaic_tile_bounding_boxes()
        metadata['bbox_all_tiles'] = aicsczi.get_all_tile_bounding_boxes()

    # get additional data by using pylibczirw directly
    metadata['pyczi_dims'] = czidoc.total_bounding_box
    metadata['pyczi_bbox_scenes'] = czidoc.scenes_bounding_rectangle
    metadata['pyczi_total_rect'] = czidoc.total_bounding_rectangle

    # check which dimension exist inside this CZI file
    metadata = checkdims_czi(czidoc, metadata)

    # determine pixel type for CZI array
    metadata['NumPy.dtype'] = {}
    for ch, px in czidoc.pixel_types.items():
        metadata['NumPy.dtype'][ch] = get_dtype_fromstring(px)

    if czidoc._is_rgb(czidoc.get_channel_pixel_type(0)):
        metadata['isRGB'] = True

    #if 'A' in aicsczi.dims:
    #    metadata['isRGB'] = True
    print('CZI is RGB :', metadata['isRGB'])

    # determine pixel type for CZI array by reading XML metadata
    try:
        metadata['PixelType'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['PixelType']
    except KeyError as e:
        print('No PixelType :', e)
        metadata['PixelType'] = None
    try:
        metadata['SizeX'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeX'])
    except KeyError as e:
        print('No X Dimension :', e)
        metadata['SizeX'] = None
    try:
        metadata['SizeY'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeY'])
    except KeyError as e:
        print('No Y Dimension :', e)
        metadata['SizeY'] = None

    try:
        metadata['SizeZ'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
    except KeyError as e:
        print('No Z Dimension :', e)
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
    except KeyError as e:
        print('No C Dimension :', e)
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    # get dimension of consitent 5D stack using AICSImageio
    #aics_img = AICSImage(filename)
    #metadata['czi_dims5D_aics'] = aics_img.dims.order
    #metadata['czi_shape5D_aics'] = aics_img.dims.shape
    #metadata['czi_dict5D_aics'] = aics_img.dims.__dict__


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
            print('Channel shortname not found :', e)
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                ['Channels']['Channel']['DyeName'])
            except KeyError as e:
                print('Channel dye not found :', e)
                channels.append('Dye-CH1')

        # get channel name
        try:
            channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                  ['Channels']['Channel']['Name'])
        except KeyError as e:
            try:
                channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                      ['Channels']['Channel']['@Name'])
            except KeyError as e:
                print('Channel name found :', e)
                channels_names.append('CH1')

        # get channel color
        try:
            channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                   ['Channels']['Channel']['Color'])
        except KeyError as e:
            print('Channel color not found :', e)
            channels_colors.append('#80808000')

    # in case of two or more channels
    if metadata['SizeC'] > 1:
        # loop over all channels
        for ch in range(metadata['SizeC']):
            # get name for dyes
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                ['Channels']['Channel'][ch]['ShortName'])
            except KeyError as e:
                print('Channel shortname not found :', e)
                try:
                    channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                    ['Channels']['Channel'][ch]['DyeName'])
                except KeyError as e:
                    print('Channel dye not found :', e)
                    channels.append('Dye-CH' + str(ch))

            # get channel names
            try:
                channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                      ['Channels']['Channel'][ch]['Name'])
            except KeyError as e:
                try:
                    channels_names.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                          ['Channels']['Channel'][ch]['@Name'])
                except KeyError as e:
                    print('Channel name not found :', e)
                    channels_names.append('CH' + str(ch))

            # get channel colors
            try:
                channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                       ['Channels']['Channel'][ch]['Color'])
            except KeyError as e:
                print('Channel color not found :', e)
                # use grayscale instead
                channels_colors.append('80808000')

    # write channels information (as lists) into metadata dictionary
    metadata['Channels'] = channels
    metadata['ChannelNames'] = channels_names
    metadata['ChannelColors'] = channels_colors

    try:
        metadata['SizeT'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeT'])
    except KeyError as e:
        print('No T Dimension :', e)
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeM'])
    except KeyError as e:
        print('No M Dimension :', e)
        if dim2none:
            metadata['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeB'])
    except KeyError as e:
        print('No B Dimension :', e)
        if dim2none:
            metadata['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
    except KeyError as e:
        print('No S Dimension :', e)
        if dim2none:
            metadata['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['SizeH'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeH'])
    except KeyError as e:
        print('No H Dimension :', e)
        if dim2none:
            metadata['SizeH'] = None
        if not dim2none:
            metadata['SizeH'] = 1

    try:
        metadata['SizeI'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeI'])
    except KeyError as e:
        print('No I Dimension :', e)
        if dim2none:
            metadata['SizeI'] = None
        if not dim2none:
            metadata['SizeI'] = 1

    try:
        metadata['SizeV'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeV'])
    except KeyError as e:
        print('No V Dimension :', e)
        if dim2none:
            metadata['SizeV'] = None
        if not dim2none:
            metadata['SizeV'] = 1

    # get the XY scaling information
    try:
        metadata['XScale'] = float(
            metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['YScale'] = float(
            metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        metadata['XScale'] = np.round(metadata['XScale'], 3)
        metadata['YScale'] = np.round(metadata['YScale'], 3)
        try:
            metadata['XScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0][
                'DefaultUnitFormat']
            metadata['YScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1][
                'DefaultUnitFormat']
        except (KeyError, TypeError) as e:
            print('Error extracting XY ScaleUnit :', e)
            metadata['XScaleUnit'] = None
            metadata['YScaleUnit'] = None
    except (KeyError, TypeError) as e:
        print('Error extracting XY Scale  :', e)

    # get the Z scaling information
    try:
        metadata['ZScale'] = float(
            metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
        metadata['ZScale'] = np.round(metadata['ZScale'], 3)
        # additional check for faulty z-scaling
        if metadata['ZScale'] == 0.0:
            metadata['ZScale'] = 1.0
        try:
            metadata['ZScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2][
                'DefaultUnitFormat']
        except (IndexError, KeyError, TypeError) as e:
            print('Error extracting Z ScaleUnit :', e)
            metadata['ZScaleUnit'] = metadata['XScaleUnit']
    except (IndexError, KeyError, TypeError) as e:
        print('Error extracting Z Scale  :', e)
        if dim2none:
            metadata['ZScale'] = None
            metadata['ZScaleUnit'] = None
        if not dim2none:
            # set to isotropic scaling if it was single plane only
            metadata['ZScale'] = metadata['XScale']
            metadata['ZScaleUnit'] = metadata['XScaleUnit']

    # convert scale unit to avoid encoding problems
    if convert_scunit:
        if metadata['XScaleUnit'] == 'µm':
            metadata['XScaleUnit'] = 'micron'
        if metadata['YScaleUnit'] == 'µm':
            metadata['YScaleUnit'] = 'micron'
        if metadata['ZScaleUnit'] == 'µm':
            metadata['ZScaleUnit'] = 'micron'

    # try to get software version
    try:
        metadata['SW-Name'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Name']
        metadata['SW-Version'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Version']
    except KeyError as e:
        print('Key not found:', e)
        metadata['SW-Name'] = None
        metadata['SW-Version'] = None

    try:
        metadata['AcqDate'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image'][
            'AcquisitionDateAndTime']
    except KeyError as e:
        print('Key not found:', e)
        metadata['AcqDate'] = None

    # check if Instrument metadata actually exist
    if pydash.objects.has(metadatadict_czi, ['ImageDocument', 'Metadata', 'Information', 'Instrument']):
        if metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument'] is not None:
            # get objective data
            if isinstance(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives'][
                              'Objective'], list):
                num_obj = len(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives'][
                                  'Objective'])
            else:
                num_obj = 1

            # if there is only one objective found
            if num_obj == 1:
                try:
                    metadata['ObjName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                               ['Instrument']['Objectives']['Objective']['Name'])
                except (KeyError, TypeError) as e:
                    print('No Objective Name :', e)
                    metadata['ObjName'].append(None)

                try:
                    metadata['ObjImmersion'] = \
                    metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives'][
                        'Objective']['Immersion']
                except (KeyError, TypeError) as e:
                    print('No Objective Immersion :', e)
                    metadata['ObjImmersion'] = None

                try:
                    metadata['ObjNA'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Objectives']['Objective']['LensNA'])
                except (KeyError, TypeError) as e:
                    print('No Objective NA :', e)
                    metadata['ObjNA'] = None

                try:
                    metadata['ObjID'] = \
                    metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives'][
                        'Objective']['Id']
                except (KeyError, TypeError) as e:
                    print('No Objective ID :', e)
                    metadata['ObjID'] = None

                try:
                    metadata['TubelensMag'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                       ['Instrument']['TubeLenses']['TubeLens']['Magnification'])
                except (KeyError, TypeError) as e:
                    print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                    metadata['TubelensMag'] = 1.0

                try:
                    metadata['ObjNominalMag'] = np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                         ['Instrument']['Objectives']['Objective'][
                                                             'NominalMagnification'])
                except (KeyError, TypeError) as e:
                    print('No Nominal Mag.:', e, 'Using Default Value = 1.0.')
                    metadata['ObjNominalMag'] = 1.0

                try:
                    if metadata['TubelensMag'] is not None:
                        metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
                    if metadata['TubelensMag'] is None:
                        print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                        metadata['ObjMag'] = metadata['ObjNominalMag'] * 1.0

                except (KeyError, TypeError) as e:
                    print('No Objective Magnification :', e)
                    metadata['ObjMag'] = None

            if num_obj > 1:
                for o in range(num_obj):

                    try:
                        metadata['ObjName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                   ['Instrument']['Objectives']['Objective'][o]['Name'])
                    except KeyError as e:
                        print('No Objective Name :', e)
                        metadata['ObjName'].append(None)

                    try:
                        metadata['ObjImmersion'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                        ['Instrument']['Objectives']['Objective'][o]['Immersion'])
                    except KeyError as e:
                        print('No Objective Immersion :', e)
                        metadata['ObjImmersion'].append(None)

                    try:
                        metadata['ObjNA'].append(np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                          ['Instrument']['Objectives']['Objective'][o]['LensNA']))
                    except KeyError as e:
                        print('No Objective NA :', e)
                        metadata['ObjNA'].append(None)

                    try:
                        metadata['ObjID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Objectives']['Objective'][o]['Id'])
                    except KeyError as e:
                        print('No Objective ID :', e)
                        metadata['ObjID'].append(None)

                    try:
                        metadata['TubelensMag'].append(
                            np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                     ['Instrument']['TubeLenses']['TubeLens'][o]['Magnification']))
                    except KeyError as e:
                        print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                        metadata['TubelensMag'].append(1.0)

                    try:
                        metadata['ObjNominalMag'].append(
                            np.float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                     ['Instrument']['Objectives']['Objective'][o]['NominalMagnification']))
                    except KeyError as e:
                        print('No Nominal Mag. :', e, 'Using Default Value = 1.0.')
                        metadata['ObjNominalMag'].append(1.0)

                    try:
                        if metadata['TubelensMag'] is not None:
                            metadata['ObjMag'].append(metadata['ObjNominalMag'][o] * metadata['TubelensMag'][o])
                        if metadata['TubelensMag'] is None:
                            print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                            metadata['ObjMag'].append(metadata['ObjNominalMag'][o] * 1.0)

                    except KeyError as e:
                        print('No Objective Magnification :', e)
                        metadata['ObjMag'].append(None)

    # get detector information

    # check if there are any detector entries inside the dictionary
    if pydash.objects.has(metadatadict_czi, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Detectors']):

        if isinstance(
                metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'],
                list):
            num_detectors = len(
                metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'])
        else:
            num_detectors = 1

        # if there is only one detector found
        if num_detectors == 1:

            # check for detector ID
            try:
                metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                              ['Instrument']['Detectors']['Detector']['Id'])
            except KeyError as e:
                print('DetectorID not found :', e)
                metadata['DetectorID'].append(None)

            # check for detector Name
            try:
                metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector']['Name'])
            except KeyError as e:
                print('DetectorName not found :', e)
                metadata['DetectorName'].append(None)

            # check for detector model
            try:
                metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Detectors']['Detector']['Manufacturer']['Model'])
            except KeyError as e:
                print('DetectorModel not found :', e)
                metadata['DetectorModel'].append(None)

            # check for detector type
            try:
                metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector']['Type'])
            except KeyError as e:
                print('DetectorType not found :', e)
                metadata['DetectorType'].append(None)

        if num_detectors > 1:
            for d in range(num_detectors):

                # check for detector ID
                try:
                    metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                  ['Instrument']['Detectors']['Detector'][d]['Id'])
                except KeyError as e:
                    print('DetectorID not found :', e)
                    metadata['DetectorID'].append(None)

                # check for detector Name
                try:
                    metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                    ['Instrument']['Detectors']['Detector'][d]['Name'])
                except KeyError as e:
                    print('DetectorName not found :', e)
                    metadata['DetectorName'].append(None)

                # check for detector model
                try:
                    metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                     ['Instrument']['Detectors']['Detector'][d]['Manufacturer'][
                                                         'Model'])
                except KeyError as e:
                    print('DetectorModel not found :', e)
                    metadata['DetectorModel'].append(None)

                # check for detector type
                try:
                    metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                    ['Instrument']['Detectors']['Detector'][d]['Type'])
                except KeyError as e:
                    print('DetectorType not found :', e)
                    metadata['DetectorType'].append(None)

    # check for well information
    metadata['Well_ArrayNames'] = []
    metadata['Well_Indices'] = []
    metadata['Well_PositionNames'] = []
    metadata['Well_ColId'] = []
    metadata['Well_RowId'] = []
    metadata['WellCounter'] = None
    metadata['SceneStageCenterX'] = []
    metadata['SceneStageCenterY'] = []

    try:
        print('Trying to extract Scene and Well information if existing ...')

        # extract well information from the dictionary
        allscenes = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes'][
            'Scene']

        # loop over all detected scenes
        for s in range(metadata['SizeS']):

            if metadata['SizeS'] == 1:
                well = allscenes
                try:
                    metadata['Well_ArrayNames'].append(allscenes['ArrayName'])
                except KeyError as e:
                    try:
                        metadata['Well_ArrayNames'].append(well['Name'])
                    except KeyError as e:
                        # print('Well Name not found :', e)
                        try:
                            metadata['Well_ArrayNames'].append(well['@Name'])
                        except KeyError as e:
                            # print('Well @Name not found :', e)
                            print('Well Name not found :', e, 'Using A1 instead')
                            metadata['Well_ArrayNames'].append('A1')

                try:
                    metadata['Well_Indices'].append(allscenes['Index'])
                except KeyError as e:
                    try:
                        metadata['Well_Indices'].append(allscenes['@Index'])
                    except KeyError as e:
                        print('Well Index not found :', e)
                        metadata['Well_Indices'].append(1)

                try:
                    metadata['Well_PositionNames'].append(allscenes['Name'])
                except KeyError as e:
                    try:
                        metadata['Well_PositionNames'].append(allscenes['@Name'])
                    except KeyError as e:
                        print('Well Position Names not found :', e)
                        metadata['Well_PositionNames'].append('P1')

                try:
                    metadata['Well_ColId'].append(np.int(allscenes['Shape']['ColumnIndex']))
                except KeyError as e:
                    print('Well ColumnIDs not found :', e)
                    metadata['Well_ColId'].append(0)

                try:
                    metadata['Well_RowId'].append(np.int(allscenes['Shape']['RowIndex']))
                except KeyError as e:
                    print('Well RowIDs not found :', e)
                    metadata['Well_RowId'].append(0)

                try:
                    # count the content of the list, e.g. how many time a certain well was detected
                    metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])
                except KeyError:
                    metadata['WellCounter'].append(Counter({'A1': 1}))

                try:
                    # get the SceneCenter Position
                    sx = allscenes['CenterPosition'].split(',')[0]
                    sy = allscenes['CenterPosition'].split(',')[1]
                    metadata['SceneStageCenterX'].append(np.double(sx))
                    metadata['SceneStageCenterY'].append(np.double(sy))
                except (TypeError, KeyError) as e:
                    print('Stage Positions XY not found :', e)
                    metadata['SceneStageCenterX'].append(0.0)
                    metadata['SceneStageCenterY'].append(0.0)

            if metadata['SizeS'] > 1:
                try:
                    well = allscenes[s]
                    metadata['Well_ArrayNames'].append(well['ArrayName'])
                except KeyError as e:
                    try:
                        metadata['Well_ArrayNames'].append(well['Name'])
                    except KeyError as e:
                        # print('Well Name not found :', e)
                        try:
                            metadata['Well_ArrayNames'].append(well['@Name'])
                        except KeyError as e:
                            # print('Well @Name not found :', e)
                            print('Well Name not found. Using A1 instead')
                            metadata['Well_ArrayNames'].append('A1')

                # get the well information
                try:
                    metadata['Well_Indices'].append(well['Index'])
                except KeyError as e:
                    try:
                        metadata['Well_Indices'].append(well['@Index'])
                    except KeyError as e:
                        print('Well Index not found :', e)
                        metadata['Well_Indices'].append(None)
                try:
                    metadata['Well_PositionNames'].append(well['Name'])
                except KeyError as e:
                    try:
                        metadata['Well_PositionNames'].append(well['@Name'])
                    except KeyError as e:
                        print('Well Position Names not found :', e)
                        metadata['Well_PositionNames'].append(None)

                try:
                    metadata['Well_ColId'].append(np.int(well['Shape']['ColumnIndex']))
                except KeyError as e:
                    print('Well ColumnIDs not found :', e)
                    metadata['Well_ColId'].append(None)

                try:
                    metadata['Well_RowId'].append(np.int(well['Shape']['RowIndex']))
                except KeyError as e:
                    print('Well RowIDs not found :', e)
                    metadata['Well_RowId'].append(None)

                # count the content of the list, e.g. how many time a certain well was detected
                metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])

                # try:
                if isinstance(allscenes, list):
                    try:
                        # get the SceneCenter Position
                        sx = allscenes[s]['CenterPosition'].split(',')[0]
                        sy = allscenes[s]['CenterPosition'].split(',')[1]
                        metadata['SceneStageCenterX'].append(np.double(sx))
                        metadata['SceneStageCenterY'].append(np.double(sy))
                    except KeyError as e:
                        print('Stage Positions XY not found :', e)
                        metadata['SceneCenterX'].append(0.0)
                        metadata['SceneCenterY'].append(0.0)
                if not isinstance(allscenes, list):
                    metadata['SceneStageCenterX'].append(0.0)
                    metadata['SceneStageCenterY'].append(0.0)

            # count the number of different wells
            metadata['NumWells'] = len(metadata['WellCounter'].keys())

    except (KeyError, TypeError) as e:
        print('No valid Scene or Well information found:', e)

    # get additional meta data about the experiment etc.
    metadata_add = get_additional_metadata_czi(metadatadict_czi)

    return metadata, metadata_add


def get_additional_metadata_czi(metadatadict_czi):
    """
    Returns a dictionary with additional CZI metadata.

    :param metadatadict_czi: complete metadata dictionary of the CZI image
    :type metadatadict_czi: dict
    :return: additional_czimd - dictionary with additional CZI metadata
    :rtype: dict
    """

    additional_czimd = {}

    try:
        additional_czimd['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except KeyError as e:
        print('Key not found :', e)
        additional_czimd['Experiment'] = None

    try:
        additional_czimd['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except KeyError as e:
        print('Key not found :', e)
        additional_czimd['HardwareSetting'] = None

    try:
        additional_czimd['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except KeyError as e:
        print('Key not found :', e)
        additional_czimd['CustomAttributes'] = None

    try:
        additional_czimd['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except KeyError as e:
        print('Key not found :', e)
        additional_czimd['DisplaySetting'] = None

    try:
        additional_czimd['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except KeyError as e:
        print('Key not found :', e)
        additional_czimd['Layers'] = None

    return additional_czimd


def checkdims_czi(czidoc, metadata):

    if 'C' in czidoc.total_bounding_box:
        metadata['hasC'] = True
    else:
        metadata['hasC'] = False

    if 'T' in czidoc.total_bounding_box:
        metadata['hasT'] = True
    else:
        metadata['hasT'] = False

    if 'Z' in czidoc.total_bounding_box:
        metadata['hasZ'] = True
    else:
        metadata['hasZ'] = False

    if 'S' in czidoc.total_bounding_box:
        metadata['hasS'] = True
    else:
        metadata['hasS'] = False

    if 'M' in czidoc.total_bounding_box:
        metadata['hasM'] = True
    else:
        metadata['hasM'] = False

    if 'B' in czidoc.total_bounding_box:
        metadata['hasB'] = True
    else:
        metadata['hasB'] = False

    if 'H' in czidoc.total_bounding_box:
        metadata['hasH'] = True
    else:
        metadata['hasH'] = False

    return metadata


def get_dtype_fromstring(pixeltype):

    dytpe = None

    if pixeltype == 'gray16' or pixeltype == 'Gray16':
        dtype = np.dtype(np.uint16)
    if pixeltype == 'gray8' or pixeltype == 'Gray8':
        dtype = np.dtype(np.uint8)
    if pixeltype == 'bgr48' or pixeltype == 'Bgr48':
        dtype = np.dtype(np.uint16)
    if pixeltype == 'bgr24' or pixeltype == 'Bgr24':
        dtype = np.dtype(np.uint8)

    return dtype


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
