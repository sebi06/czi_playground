# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_tools.py
# Version     : 0.0.5
# Author      : sebi06
# Date        : 18.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations

import pylibCZIrw.czi
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da


def read_7darray(filename: str) -> np.ndarray:

    # get the complete metadata at once as one big class
    mdata = czimd.get_czimetadata_extended(filename)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        if mdata.image.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent
            sizeX = mdata.bbox.all_scenes[0].w
            sizeY = mdata.bbox.all_scenes[0].h

        if mdata.image.SizeS is None:
            sizeX = mdata.image.SizeX
            sizeY = mdata.image.SizeY

        # check if dimensions are None (because they do not exist for that image)
        sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        # define the dimension order to be STZCYXA
        array7d = np.empty([sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # read array for the scene
        for s, t, z, c in product(range(sizeS),
                                  range(sizeT),
                                  range(sizeZ),
                                  range(sizeC)):
            if mdata.image.SizeS is None:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

            # check if the image2d is really not too big
            if (mdata.bbox.total_bounding_box["X"][1] > mdata.image.SizeX or
                    mdata.bbox.total_bounding_box["Y"][1] > mdata.image.SizeY):

                image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

            # array6d[s, t, z, c, ...] = image2d[..., 0]
            array7d[s, t, z, c, ...] = image2d

    return array7d


def read_7darray_lazy(filename: str) -> da.Array:

    def read_scene6d(filename: str,
                     sizes: Tuple[int, int, int, int, int],
                     s: int,
                     mdata: pylibCZIrw.czi_metadata.CziMetadata):

        # define the dimension order to be TZCYXA
        array6d = da.empty([sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], 3 if mdata.isRGB else 1],
                           dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

            # read array for the scene
            for t, z, c in product(range(sizeT),
                                   range(sizeZ),
                                   range(sizeC)):

                if mdata.image.SizeS is None:
                    image2d = czidoc.read()
                else:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

                # check if the image2d is really not too big
                if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                    image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                array6d[t, z, c, ...] = image2d

        return array6d

    # get the metadata
    mdata = get_czimdata_extended(filename)

    if mdata.image.SizeS is not None:
        # get size for a single scene using the 1st
        # works only if scene shape is consistent
        sizeX = mdata.bbox.all_scenes[0].w
        sizeY = mdata.bbox.all_scenes[0].h

    if mdata.image.SizeS is None:
        sizeX = mdata.image.SizeX
        sizeY = mdata.image.SizeY

    # check if dimensions are None (because they do not exist for that image)
    sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

    sizes = (sizeT, sizeZ, sizeC, sizeY, sizeX)

    # define the required shape
    sp = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_scene6d)  # lazy reader
    lazy_arrays = [lazy_process_image(filename, sizes, s, mdata) for s in range(sizeS)]

    dask_arrays = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array7d = da.stack(dask_arrays, axis=0)

    return array7d
