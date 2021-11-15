from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product

def read_6darray(filename: str) -> np.ndarray:
    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:
        #czidoc = pyczi.open_czi(filename)

        if mdata.dims.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent
            sizeX = mdata.bbox.all_scenes[0].w
            sizeY = mdata.bbox.all_scenes[0].h

        if mdata.dims.SizeS is None:
            sizeX = mdata.dims.SizeX
            sizeY = mdata.dims.SizeY

        if mdata.dims.SizeC is None:
            sizeC = 1
        else:
            sizeC = mdata.dims.SizeC

        if mdata.dims.SizeZ is None:
            sizeZ = 1
        else:
            sizeZ = mdata.dims.SizeZ

        if mdata.dims.SizeT is None:
            sizeT = 1
        else:
            sizeT = mdata.dims.SizeT

        if mdata.dims.SizeS is None:
            sizeS = 1
        else:
            sizeS = mdata.dims.SizeS

        # define the dimension order to be STZCYX
        array6d = np.empty([sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # read array for the scene
        for s, t, z, c in product(range(sizeS),
                                  range(sizeT),
                                  range(sizeZ),
                                  range(sizeC)):
            if mdata.dims.SizeS is None:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

            #array6d[s, t, z, c, ...] = image2d[..., 0]
            array6d[s, t, z, c, ...] = image2d

    return array6d