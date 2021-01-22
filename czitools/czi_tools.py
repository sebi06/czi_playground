# -*- coding: utf-8 -*-

#################################################################
# File        : czi_tools.py
# Version     : 0.0.5
# Author      : czsrh
# Date        : 22.01.2021
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2021 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import os
from aicsimageio import AICSImage, imread, imread_dask
import aicspylibczi
import imgfileutils as imf
import itertools as it
from tqdm import tqdm, trange
from tqdm.contrib.itertools import product
import nested_dict as nd
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser as dt
from lxml import etree
import progressbar


def define_czi_planetable():

    df = pd.DataFrame(columns=['Subblock',
                               'Scene',
                               'Tile',
                               'T',
                               'Z',
                               'C',
                               'X[micron]',
                               'Y[micron]',
                               'Z[micron]',
                               'Time[s]',
                               'xstart',
                               'ystart',
                               'xwidth',
                               'ywidth'])

    return df


def get_czi_planetable(czifile):

    # get the czi object using pylibczi
    czi = aicspylibczi.CziFile(czifile)

    # get the czi metadata
    md, add = imf.get_metadata(czifile)

    # initialize the plane table
    df_czi = define_czi_planetable()

    # define subblock counter
    sbcount = -1

    # create progressbar
    #total = md['SizeS'] * md['SizeM'] * md['SizeT'] * md['SizeZ'] * md['SizeC']
    #pbar = tqdm(total=total)

    #pbar = progressbar.ProgressBar(max_value=total)
    # in case the CZI has the M-Dimension
    if md['czi_isMosaic']:

        for s, m, t, z, c in product(range(md['SizeS']),
                                     range(md['SizeM']),
                                     range(md['SizeT']),
                                     range(md['SizeZ']),
                                     range(md['SizeC'])):

            sbcount += 1
            # print(s, m, t, z, c)
            info = czi.read_subblock_rect(S=s, M=m, T=t, Z=z, C=c)

            # read information from subblock
            sb = czi.read_subblock_metadata(unified_xml=True,
                                            B=0,
                                            S=s,
                                            M=m,
                                            T=t,
                                            Z=z,
                                            C=c)

            try:
                time = sb.xpath('//AcquisitionTime')[0].text
                timestamp = dt.parse(time).timestamp()
            except IndexError as e:
                timestamp = 0.0

            try:
                xpos = np.double(sb.xpath('//StageXPosition')[0].text)
            except IndexError as e:
                xpos = 0.0

            try:
                ypos = np.double(sb.xpath('//StageYPosition')[0].text)
            except IndexError as e:
                ypos = 0.0

            try:
                zpos = np.double(sb.xpath('//FocusPosition')[0].text)
            except IndexError as e:
                zpos = 0.0

            df_czi = df_czi.append({'Subblock': sbcount,
                                    'Scene': s,
                                    'Tile': m,
                                    'T': t,
                                    'Z': z,
                                    'C': c,
                                    'X[micron]': xpos,
                                    'Y[micron]': ypos,
                                    'Z[micron]': zpos,
                                    'Time[s]': timestamp,
                                    'xstart': info[0],
                                    'ystart': info[1],
                                    'xwidth': info[2],
                                    'ywidth': info[3]},
                                   ignore_index=True)

    if not md['czi_isMosaic']:

        """
        for s, t, z, c in it.product(range(md['SizeS']),
                                     range(md['SizeT']),
                                     range(md['SizeZ']),
                                     range(md['SizeC'])):
        """

        for s, t, z, c in product(range(md['SizeS']),
                                  range(md['SizeT']),
                                  range(md['SizeZ']),
                                  range(md['SizeC'])):

            sbcount += 1
            info = czi.read_subblock_rect(S=s, T=t, Z=z, C=c)

            # read information from subblocks
            sb = czi.read_subblock_metadata(unified_xml=True, B=0, S=s, T=t, Z=z, C=c)

            try:
                time = sb.xpath('//AcquisitionTime')[0].text
                timestamp = dt.parse(time).timestamp()
            except IndexError as e:
                timestamp = 0.0

            try:
                xpos = np.double(sb.xpath('//StageXPosition')[0].text)
            except IndexError as e:
                xpos = 0.0

            try:
                ypos = np.double(sb.xpath('//StageYPosition')[0].text)
            except IndexError as e:
                ypos = 0.0

            try:
                zpos = np.double(sb.xpath('//FocusPosition')[0].text)
            except IndexError as e:
                zpos = 0.0

            df_czi = df_czi.append({'Subblock': sbcount,
                                    'Scene': s,
                                    'Tile': 0,
                                    'T': t,
                                    'Z': z,
                                    'C': c,
                                    'X[micron]': xpos,
                                    'Y[micron]': ypos,
                                    'Z[micron]': zpos,
                                    'Time[s]': timestamp,
                                    'xstart': info[0],
                                    'ystart': info[1],
                                    'xwidth': info[2],
                                    'ywidth': info[3]},
                                   ignore_index=True)

    # normalize timestamps
    df_czi = imf.norm_columns(df_czi, colname='Time[s]', mode='min')

    # cast data  types
    df_czi = df_czi.astype({'Subblock': 'int32',
                            'Scene': 'int32',
                            'Tile': 'int32',
                            'T': 'int32',
                            'Z': 'int32',
                            'C': 'int16',
                            'xstart': 'int32',
                            'xstart': 'int32',
                            'ystart': 'int32',
                            'xwidth': 'int32',
                            'ywidth': 'int32'},
                           copy=False,
                           errors='ignore')

    return df_czi


def save_planetable(df, filename, separator=',', index=True):
    """Save dataframe as CSV table

    :param df: Dataframe to be saved as CSV.
    :type df: pd.DataFrame
    :param filename: filename of the CSV to be written
    :type filename: str
    :param separator: seperator for the CSV file, defaults to ','
    :type separator: str, optional
    :param index: option write the index into the CSV file, defaults to True
    :type index: bool, optional
    :return: filename of the csvfile that was written
    :rtype: str
    """
    csvfile = os.path.splitext(filename)[0] + '_planetable.csv'

    # write the CSV data table
    df.to_csv(csvfile, sep=separator, index=index)

    return csvfile


def filterplanetable(planetable, S=0, T=0, Z=0, C=0):

    # filter planetable for specific scene
    if S > planetable['Scene'].max():
        print('Scene Index was invalid. Using Scene = 0.')
        S = 0
    pt = planetable[planetable['Scene'] == S]

    # filter planetable for specific timepoint
    if T > planetable['T'].max():
        print('Time Index was invalid. Using T = 0.')
        T = 0
    pt = planetable[planetable['T'] == T]

    # filter resulting planetable pt for a specific z-plane
    if Z > planetable['Z[micron]'].max():
        print('Z-Plane Index was invalid. Using Z = 0.')
        zplane = 0
    pt = pt[pt['Z[micron]'] == Z]

    # filter planetable for specific channel
    if C > planetable['C'].max():
        print('Channel Index was invalid. Using C = 0.')
        C = 0
    pt = planetable[planetable['C'] == C]

    # return filtered planetable
    return pt


def get_bbox_scene(cziobject, sceneindex=0):
    """Get the min / max extend of a given scene from a CZI mosaic image
    at pyramid level = 0 (full resolution)

    :param czi: CZI object for from aicspylibczi
    :type czi: Zeiss CZI file object
    :param sceneindex: index of the scene, defaults to 0
    :type sceneindex: int, optional
    :return: tuple with (XSTART, YSTART, WIDTH, HEIGHT) extend in pixels
    :rtype: tuple
    """

    # get all bounding boxes
    bboxes = cziobject.mosaic_scene_bounding_boxes(index=sceneindex)

    # initialize lists for required values
    xstart = []
    ystart = []
    tilewidth = []
    tileheight = []

    # loop over all tiles for the specified scene
    for box in bboxes:

        # get xstart, ystart amd tile widths and heights
        xstart.append(box[0])
        ystart.append(box[1])
        tilewidth.append(box[2])
        tileheight.append(box[3])

    # get bounding box for the current scene
    XSTART = min(xstart)
    YSTART = min(ystart)

    # do not forget to add the width and height of the last tile :-)
    WIDTH = max(xstart) - XSTART + tilewidth[-1]
    HEIGHT = max(ystart) - YSTART + tileheight[-1]

    return XSTART, YSTART, WIDTH, HEIGHT


def read_scene_bbox(cziobject, metadata,
                    sceneindex=0,
                    channel=0,
                    timepoint=0,
                    zplane=0,
                    scalefactor=1.0):
    """Read a specific scene from a CZI image file.

    :param cziobject: The CziFile reader object from aicspylibczi
    :type cziobject: CziFile
    :param metadata: Image metadata dictionary from imgfileutils
    :type metadata: dict
    :param sceneindex: Index of scene, defaults to 0
    :type sceneindex: int, optional
    :param channel: Index of channel, defaults to 0
    :type channel: int, optional
    :param timepoint: Index of Timepoint, defaults to 0
    :type timepoint: int, optional
    :param zplane: Index of z-plane, defaults to 0
    :type zplane: int, optional
    :param scalefactor: scaling factor to read CZI image pyramid, defaults to 1.0
    :type scalefactor: float, optional
    :return: scene as a numpy array
    :rtype: NumPy.Array
    """
    # set variables
    scene = None
    hasT = False
    hasZ = False

    # check if scalefactor has a reasonable value
    if scalefactor < 0.01 or scalefactor > 1.0:
        print('Scalefactor too small or too large. Will use 1.0 as fallback')
        scalefactor = 1.0

    # check if CZI has T or Z dimension
    if 'T' in metadata['dims_aicspylibczi']:
        hasT = True
    if 'T' in metadata['dims_aicspylibczi']:
        hasZ = True

    # get the bounding box for the specified scene
    xmin, ymin, width, height = get_bbox_scene(cziobject,
                                               sceneindex=sceneindex)

    # read the scene as numpy array using the correct function calls
    if hasT is True and hasZ is True:
        scene = cziobject.read_mosaic(region=(xmin, ymin, width, height),
                                      scale_factor=scalefactor,
                                      T=timepoint,
                                      Z=zplane,
                                      C=channel)

    if hasT is True and hasZ is False:
        scene = cziobject.read_mosaic(region=(xmin, ymin, width, height),
                                      scale_factor=scalefactor,
                                      T=timepoint,
                                      C=channel)

    if hasT is False and hasZ is True:
        scene = cziobject.read_mosaic(region=(xmin, ymin, width, height),
                                      scale_factor=scalefactor,
                                      Z=zplane,
                                      C=channel)

    if hasT is False and hasZ is False:
        scene = cziobject.read_mosaic(region=(xmin, ymin, width, height),
                                      scale_factor=scalefactor,
                                      C=channel)

    # add new entries to metadata to adjust XY scale due to scaling factor
    metadata['XScale Pyramid'] = metadata['XScale'] * 1 / scalefactor
    metadata['YScale Pyramid'] = metadata['YScale'] * 1 / scalefactor

    return scene, (xmin, ymin, width, height), metadata
