import dask.array
import pylibCZIrw.czi_metadata
from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import napari
from czitools import pylibczirw_tools
from czitools import misc, napari_tools
import numpy as np
from czitools import pylibczirw_metadata as czimd
from aicspylibczi import CziFile
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def read_czi_lazy(filename):

    # get the metadata
    mdata = pylibczirw_tools.get_czimdata_extended(filename)

    if mdata.dims.SizeS is not None:
        # get size for a single scene using the 1st
        # works only if scene shape is consistent
        sizeX = mdata.bbox.all_scenes[0].w
        sizeY = mdata.bbox.all_scenes[0].h

    if mdata.dims.SizeS is None:
        sizeX = mdata.dims.SizeX
        sizeY = mdata.dims.SizeY

    # check if dimensions are None (because they do not exist for that image)
    sizeC = misc.check_dimsize(mdata.dims.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.dims.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.dims.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.dims.SizeS, set2value=1)

    def read7d(filename, mdata):

        # define the dimension order to be STZCYXA
        array7d = np.empty([sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1], dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

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
                if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                    image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                # array6d[s, t, z, c, ...] = image2d[..., 0]
                array7d[s, t, z, c, ...] = image2d

        return array7d


##################################################

filename = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"

# get the complete metadata at once as one big class
mdata_extended = get_czimetadata_extended(filename)

# define the required shape
sp = [sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]
lazy_readczi = dask.delayed(readczi)
lazy_arrays = [lazy_readczi(mdata, czidoc, has_scene, s, t, z, c) for s, t, z, c in product(range(sizeS),
                                                                                            range(sizeT),
                                                                                            range(sizeZ),
                                                                                            range(sizeC))]

dask_arrays = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

# remove A dimension do display the array inside Napari
mdarray = np.squeeze(mdarray, axis=-1)
mdata.dimstring = "STZCYX"
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(mdata.dimstring)
setattr(mdata, "dim_order", dim_order)
setattr(mdata, "dim_index", dim_index)
setattr(mdata, "dim_valid", dim_valid)

# define the dimension string
print(mdarray.shape)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, mdarray, mdata,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()


# def cziread_STZCYX(filename: str) -> da.Array:
#
#     def read_CYX(czidoc: pyczi.CziReader,
#                  has_scene: bool = False,
#                  s: int = 0,
#                  t: int = 0,
#                  z: int = 0,
#                  c: int = 0) -> np.ndarray:
#
#         if not has_scene:
#             arrayYX = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
#         elif hast_scene:
#             arrayYX = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)
#
#         return arrayYX
#
#     # get the metadata
#     mdata = get_czimdata_extended(filename)
#
#     if mdata.image.SizeS is not None:
#         # get size for a single scene using the 1st
#         # works only if scene shape is consistent
#         sizeX = mdata.bbox.all_scenes[0].w
#         sizeY = mdata.bbox.all_scenes[0].h
#         has_scene = True
#
#     if mdata.dims.SizeS is None:
#         sizeX = mdata.image.SizeX
#         sizeY = mdata.image.SizeY
#         has_scene = False
#
#     # check if dimensions are None (because they do not exist for that image)
#     sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
#     sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
#     sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
#     sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)
#
#     #lazy_STZCYX = dask.delayed(read_STZCYX)
#     #lazy_TZCYX = dask.delayed(read_TZCYX)
#     #lazy_ZCYX = dask.delayed(read_ZCYX)
#     #lazy_CYX = dask.delayed(read_CYX)
#
#     # read channels lazy
#
#     #sp_TZCYX = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]
#     #sp_ZCYX = [sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]
#     #sp_CYX = [sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]
#
#     # define the required shape
#     sp = [sizeS, sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]
#
#     lazy_arrays_CXY = [lazy_readczi(czidoc, has_scene, s, t, z, c) for s, t, z, c in product(range(sizeS),
#                                                                                              range(sizeT),
#                                                                                              range(sizeZ),
#                                                                                              range(sizeC))]
#
#     dask_arrays_CXY = [da.from_delayed(lazy_array, shape=sp_CYX, dtype=mdata.npdtype) for lazy_array in lazy_arrays]
#     #array_lazy_CXY = da.stack(dask_arrays_C, axis=0)
#
#     # read array for the scene
#     for t, z, c in product(range(sizeT),
#                            range(sizeZ),
#                            range(sizeC)):
#
#         # read channels
#         dask_arrays_CXY = [da.from_delayed(lazy_array, shape=sp_CYX, dtype=mdata.npdtype) for lazy_array in lazy_arrays]
#         array_lazy_CXY = da.stack(dask_arrays_C, axis=0)
