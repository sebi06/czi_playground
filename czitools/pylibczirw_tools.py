# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_tools.py
# Version     : 0.0.3
# Author      : sebi06
# Date        : 14.12.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da
from dask import delayed


def read_7darray(filename: str) -> np.ndarray:

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        if mdata.dims.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent
            sizeX = mdata.bbox.all_scenes[0].w
            sizeY = mdata.bbox.all_scenes[0].h

        if mdata.dims.SizeS is None:
            sizeX = mdata.dims.SizeX
            sizeY = mdata.dims.SizeY

        # check if dimensions are None (because the do not exist for that image)
        sizeC = misc.check_dimsize(mdata.dims.SizeC, set2value=1)
        sizeZ = misc.check_dimsize(mdata.dims.SizeZ, set2value=1)
        sizeT = misc.check_dimsize(mdata.dims.SizeT, set2value=1)
        sizeS = misc.check_dimsize(mdata.dims.SizeS, set2value=1)

        # define the dimension order to be STZCYXA
        array7d = np.empty([sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # read array for the scene
        for s, t, z, c in product(range(sizeS),
                                  range(sizeT),
                                  range(sizeZ),
                                  range(sizeC)):
            if mdata.dims.SizeS is None:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

            # check if the image2d is really not too big
            if mdata.pyczi_dims["X"][1] > mdata.dims.SizeX or mdata.pyczi_dims["Y"][1] > mdata.dims.SizeY:
                image2d = image2d[..., 0:mdata.dims.SizeY, 0:mdata.dims.SizeX, :]

            # array6d[s, t, z, c, ...] = image2d[..., 0]
            array7d[s, t, z, c, ...] = image2d

    return array7d


def read_7darray_lazy(filename: str) -> da.array:

    def read_scene6d(filename, s, mdata):

        # define the dimension order to be TZCYXA
        array6d = np.empty([sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

            # read array for the scene
            for t, z, c in product(range(sizeT),
                                   range(sizeZ),
                                   range(sizeC)):

                if mdata.dims.SizeS is None:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
                else:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

                # check if the image2d is really not too big
                if mdata.pyczi_dims["X"][1] > mdata.dims.SizeX or mdata.pyczi_dims["Y"][1] > mdata.dims.SizeY:
                    image2d = image2d[..., 0:mdata.dims.SizeY, 0:mdata.dims.SizeX, :]

                array6d[t, z, c, ...] = image2d

        return array6d

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    if mdata.dims.SizeS is not None:
        # get size for a single scene using the 1st
        # works only if scene shape is consistent
        sizeX = mdata.bbox.all_scenes[0].w
        sizeY = mdata.bbox.all_scenes[0].h

    if mdata.dims.SizeS is None:
        sizeX = mdata.dims.SizeX
        sizeY = mdata.dims.SizeY

    # check if dimensions are None (because the do not exist for that image)
    sizeC = misc.check_dimsize(mdata.dims.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.dims.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.dims.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.dims.SizeS, set2value=1)

    # define the required shape
    sp = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_scene6d)  # lazy reader
    lazy_arrays = [lazy_process_image(filename, s, mdata) for s in range(sizeS)]

    dask_arrays = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array7d = da.stack(dask_arrays, axis=0)

    return array7d


def read_7darray_lazy_st(filename: str) -> da.array:

    def read_scene5d(filename, s, t, mdata):

        # define the dimension order to be TZCYXA
        array5d = np.empty([sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

            # read array for the scene
            for z, c in product(range(sizeZ),
                                range(sizeC)):

                if mdata.dims.SizeS is None:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
                else:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

                # check if the image2d is really not too big
                if mdata.pyczi_dims["X"][1] > mdata.dims.SizeX or mdata.pyczi_dims["Y"][1] > mdata.dims.SizeY:
                    image2d = image2d[..., 0:mdata.dims.SizeY, 0:mdata.dims.SizeX, :]

                array5d[z, c, ...] = image2d

        return array5d

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    if mdata.dims.SizeS is not None:
        # get size for a single scene using the 1st
        # works only if scene shape is consistent
        sizeX = mdata.bbox.all_scenes[0].w
        sizeY = mdata.bbox.all_scenes[0].h

    if mdata.dims.SizeS is None:
        sizeX = mdata.dims.SizeX
        sizeY = mdata.dims.SizeY

    # check if dimensions are None (because the do not exist for that image)
    sizeC = misc.check_dimsize(mdata.dims.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.dims.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.dims.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.dims.SizeS, set2value=1)

    # define the required shape
    sp = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_scene5d)  # lazy reader
    lazy_arrays = [lazy_process_image(filename, s, mdata) for s in range(sizeS)]

    dask_arrays = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array7d = da.stack(dask_arrays, axis=0)

    return array7d
