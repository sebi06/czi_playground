# -*- coding: utf-8 -*-

#################################################################
# File        : test_aicsimageio.py
# Version     : 0.0.2
# Author      : sebi06
# Date        : 14.12.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import napari
from czitools import czi_metadata as czimd_aics
from czitools import pylibczirw_metadata as czimd

from aicsimageio import AICSImage
from czitools import misc, napari_tools

filename = r'd:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'

# get the complete metadata at once as one big class
mdata = czimd_aics.CziMetadata(filename)

# test using AICSImageIO
aics_img = AICSImage(filename)
print(aics_img.shape)
for k, v in aics_img.dims.items():
    print(k, v)

# get the stack as dask array
stack = misc.get_daskstack(aics_img)

mdata.dimstring = "S" + aics_img.dims.order
dim_order, dim_index, dim_valid = czimd_aics.CziMetadata.get_dimorder(mdata.dimstring)
setattr(mdata, "dim_order", dim_order)
setattr(mdata, "dim_index", dim_index)
setattr(mdata, "dim_valid", dim_valid)

# start the napari viewer and show the image
# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, stack, mdata,
                           blending="additive",
                           contrast="napari_auto",
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
