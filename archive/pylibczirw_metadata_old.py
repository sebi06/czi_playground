# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_metadata_old.py
# Version     : 0.0.7
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


def get_czimetadata_extended(filename: str) -> Type[pyczi.CziMetadata]:

    with pyczi.open_czi(filename) as czidoc:
        mdata = pyczi.CziMetadata(czidoc.raw_metadata)

        mdata.__setattr__("metadict", czidoc.metadata)

        # get the bounding boxes
        mdata.__setattr__("bbox", CziBoundingBox(filename))

        # get the pixel typed for all channels
        mdata.__setattr__("pixeltypes", czidoc.pixel_types)
        mdata.__setattr__("isRGB", czidoc._is_rgb(czidoc.pixel_types[0]))

        # determine pixel type for CZI array
        pixeltype = czidoc.pixel_types[0]
        npdtype, maxvalue = get_dtype_fromstring(pixeltype)
        mdata.__setattr__("npdtype", npdtype)

        # try to guess if the CZI is a mosaic file
        if mdata.image.SizeM is None or mdata.image.SizeM == 1:
            mdata.__setattr__("isMosaic", False)
        else:
            mdata.__setattr__("isMosaic", True)

        mdata.scale.__setattr__("ratio", get_scale_ratio(scalex=mdata.scale.X,
                                                         scaley=mdata.scale.Y,
                                                         scalez=mdata.scale.Z))

        # get some additional metadata using aipylibczi
        try:
            from aicspylibczi import CziFile

            # get the general CZI object using aicspylibczi
            aicsczi = CziFile(filename)
            # get additional data by using aicspylibczi directly
            mdata.__setattr__("dimstring", aicsczi.dims)
            mdata.__setattr__("dims_shape", aicsczi.get_dims_shape())
            mdata.__setattr__("size", aicsczi.size)
            mdata.__setattr__("isMosaic", aicsczi.is_mosaic())

            dim_order, dim_index, dim_valid = get_dimorder(aicsczi.dims)
            mdata.__setattr__("dim_order", dim_order)
            mdata.__setattr__("dim_valid", dim_valid)
            mdata.__setattr__("dim_index", dim_index)

        except ImportError as e:
            print(e)

    return mdata


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
