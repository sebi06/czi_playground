# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Version     : 0.0.3
# Author      : sebi06
# Date        : 15.10.2021
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from tkinter import filedialog
from tkinter import *
import zarr
import pandas as pd
import dask
import dask.array as da
import numpy as np
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from aicspylibczi import CziFile
from aicsimageio import AICSImage
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def openfile(directory: str,
             title: str = "Open CZI Image File",
             ftypename: str = "CZI Files",
             extension: str = "*.czi") -> str:
    """ Open a simple Tk dialog to select a file.

    :param directory: default directory
    :param title: title of the dialog window
    :param ftypename: name of allowed file type
    :param extension: extension of allowed file type
    :return: filepath object for the selected
    """

    # request input and output image path from user
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfile(title=title,
                                        initialdir=directory,
                                        filetypes=[(ftypename, extension)])
    if input_path is not None:
        return input_path.name
    if input_path is None:
        return ""


def slicedim(array: Union[np.ndarray, dask.array.Array, zarr.Array],
             dimindex: int,
             posdim: int) -> np.ndarray:
    """Slice out a specific dimension without (!) dropping the dimension
    of the array to conserve the dimorder string
    this should work for Numpy.Array, Dask and ZARR ...

    :param array: input array
    :param dimindex: index of the slice dimension to be kept
    :param posdim: position of the dimension to be sliced
    :return: sliced array
    """

    #if posdim == 0:
    #    array_sliced = array[dimindex:dimindex + 1, ...]
    #if posdim == 1:
    #    array_sliced = array[:, dimindex:dimindex + 1, ...]
    #if posdim == 2:
    #    array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    #if posdim == 3:
    #    array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    #if posdim == 4:
    #    array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    #if posdim == 5:
    #    array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(data: np.ndarray,
                 corr_min: float = 1.0,
                 offset_min: int = 0,
                 corr_max: float = 0.85,
                 offset_max: int = 0) -> Tuple[int, int]:
    """Calculate the scaling for better display

    :param data: Calculate min / max scaling
    :type data: Numpy.Array or dask.array or zarr.array
    :param corr_min: correction factor for minvalue, defaults to 1.0
    :type corr_min: float, optional
    :param offset_min: offset for min value, defaults to 0
    :type offset_min: int, optional
    :param corr_max: correction factor for max value, defaults to 0.85
    :type corr_max: float, optional
    :param offset_max: offset for max value, defaults to 0
    :type offset_max: int, optional
    :return: list with [minvalue, maxvalue]
    :rtype: list
    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # use dask.compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data), np.max(data)

    end = time.time()

    minvalue = np.round((minvalue + offset_min) * corr_min, 0)
    maxvalue = np.round((maxvalue + offset_max) * corr_max, 0)

    print("Scaling:", minvalue, maxvalue)
    print("Calculation of Min-Max [s] : ", end - start)

    return minvalue, maxvalue


def md2dataframe(md_dict: Dict,
                 paramcol: str = "Parameter",
                 keycol: str = "Value") -> pd.DataFrame:
    """Convert the metadata dictionary to a Pandas DataFrame.

    :param metadata: MeteData dictionary
    :type metadata: dict
    :param paramcol: Name of Columns for the MetaData Parameters, defaults to "Parameter"
    :type paramcol: str, optional
    :param keycol: Name of Columns for the MetaData Values, defaults to "Value"
    :type keycol: str, optional
    :return: Pandas DataFrame containing all the metadata
    :rtype: Pandas.DataFrame
    """
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in md_dict.keys():
        d = {"Parameter": k, "Value": md_dict[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def sort_dict_by_key(unsorted_dict: Dict) -> Dict:
    sorted_keys = sorted(unsorted_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict.update({key: unsorted_dict[key]})

    return sorted_dict


def writexml_czi(filename: str, xmlsuffix: str = '_CZI_MetaData.xml') -> str:
    """Write XML information of CZI to disk

    :param filename: CZI image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    # get metadata dictionary using aicspylibczi
    aicsczi = CziFile(filename)
    metadata_xmlstr = ET.tostring(aicsczi.meta)

    # change file name
    xmlfile = filename.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(metadata_xmlstr))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile


def addzeros(number: int) -> str:
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


def get_fname_woext(filepath: str) -> str:
    """Get the complete path of a file without the extension
    It also will works for extensions like c:\myfile.abc.xyz
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

    # remove real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def check_dimsize(mdata: Union[int, None], set2one: int = 1) -> int:

    # check if the dimension entry is None
    if mdata is None:
        size = 1
    if mdata is not None:
        size = mdata

    return size


def get_daskstack(aics_img: AICSImage) -> List:

    stacks = []
    for scene in aics_img.scenes:
        aics_img.set_scene(scene)
        stacks.append(aics_img.dask_data)

    stacks = da.stack(stacks)

    return stacks
