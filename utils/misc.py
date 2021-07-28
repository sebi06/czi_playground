# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Version     : 0.0.1
# Author      : sebi06
# Date        : 28.07.2021
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
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def openfile(directory: str,
             title: str = "Open CZI Image File",
             ftypename: str = "CZI Files",
             extension: str = "*.czi") -> str:

    # request input and output image path from user
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfile(title=title,
                                        initialdir=directory,
                                        filetypes=[(ftypename, extension)])
    if input_path is not None:
        return input_path.name
    if input_path is None:
        return ''


def slicedim(array: Union[np.ndarray, dask.array.Array, zarr.Array],
             dimindex: int,
             posdim: int) -> np.ndarray:
    """slice out a specific channel without (!) dropping the dimension
    of the array to conserve the dimorder string
    this should work for Numpy.Array, Dask and ZARR ...

    if posdim == 0:
        array_sliced = array[dimindex:dimindex + 1, ...]
    if posdim == 1:
        array_sliced = array[:, dimindex:dimindex + 1, ...]
    if posdim == 2:
        array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    if posdim == 3:
        array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    if posdim == 4:
        array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    if posdim == 5:
        array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]
    """

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
        minvalue, maxvalue = np.min(data, initial=0), np.max(data, initial=0)

    end = time.time()

    minvalue = np.round((minvalue + offset_min) * corr_min, 0)
    maxvalue = np.round((maxvalue + offset_max) * corr_max, 0)

    print('Scaling:', minvalue, maxvalue)
    print('Calculation of Min-Max [s] : ', end - start)

    return minvalue, maxvalue


def md2dataframe(md_dict: Dict,
                 paramcol: str = 'Parameter',
                 keycol: str = 'Value') -> pd.DataFrame:
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

    for k in md_dict.keys():
        d = {'Parameter': k, 'Value': md_dict[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe
