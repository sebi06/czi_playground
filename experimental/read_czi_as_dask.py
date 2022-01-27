from pylibCZIrw import czi as pyczi
import numpy as np
from czitools import pylibczirw_metadata as czimd
from czitools import misc
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da


def read_czi_lazy(filename, remove_Adim=True):

    def read_4d(filename: str,
                sizes: Tuple[int, int, int, int],
                s: int,
                t: int,
                mdata: czimd.CziMetadata,
                remove_Adim: bool = True) -> np.ndarray:

        # array dimension will be ZCYX(A)
        array_ZCYX = da.empty([sizes[0], sizes[1], sizes[2], sizes[3], 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

            # read array for the ZCYX
            for z, c in product(range(sizes[0]),
                                range(sizes[1])):

                if mdata.image.SizeS is None:
                    image2d = czidoc.read()
                else:
                    image2d = czidoc.read(plane={"T": t, "Z": z, "C": c}, scene=s)

                # check if the image2d is really not too big
                if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                    image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                array_ZCYX[z, c, ...] = image2d

        if remove_Adim:
            array_ZCYX = np.squeeze(array_ZCYX, axis=-1)

        return array_ZCYX

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    if mdata.image.SizeS is not None:
        # get size for a single scene using the 1st
        # works only if scene shape is consistent
        sizeX = mdata.bbox.all_scenes[0].w
        sizeY = mdata.bbox.all_scenes[0].h

    if mdata.image.SizeS is None:
        sizeX = mdata.dims.SizeX
        sizeY = mdata.dims.SizeY

    # check if dimensions are None (because they do not exist for that image)
    sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

    sizes = (sizeZ, sizeC, sizeY, sizeX)

    # define the required shape
    if remove_Adim:
        sp = [sizeZ, sizeC, sizeY, sizeX]
    if not remove_Adim:
        sp = [sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_4d)  # lazy reader
    lazy_arrays = [lazy_process_image(filename, sizes, s, t, mdata, remove_Adim) for s in range(sizeT)]

    dask_arrays_ZCYX = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array_TZCYX = da.stack(dask_arrays_ZCYX, axis=0)

    for s in tqdm

    if remove_Adim:
        dimstring = "STZCYX"
