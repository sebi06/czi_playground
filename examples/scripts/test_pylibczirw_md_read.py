# -*- coding: utf-8 -*-

#################################################################
# File        : test_pylibczirw_md_read.py
# Version     : 0.0.1
# Author      : sebi06
# Date        : 24.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from dataclasses import dataclass
from czitools import pylibczirw_metadata as czimd
from czitools import pylibczirw_tools
from czitools import napari_tools
import napari
import numpy as np
from pylibCZIrw import czi as pyczi


# filename = r'C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
# filename = r"C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
# filename = r'C:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
# filename = r"C:\Testdata_Zeiss\CZI_Testfiles\tobacco_z=10_tiles.czi"
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"
filename = r"D:\Testdata_Zeiss\CZI_Testfiles\demo_read.czi"
# filename = r"/datadisk1/testpictures/Testdata_Zeiss/CZI_Testfiles/CellDivision_T=10_Z=20_CH=1_DCV.czi"
# filename = r"/datadisk1/testpictures/Testdata_Zeiss/CZI_Testfiles/CellDivision_T=10_Z=15_CH=2_DCV_small.czi"


# get the raw metadata as a XML or dictionary
with pyczi.open_czi(filename) as czidoc:
    mdata_comp_xml = czidoc.raw_metadata
    mdata_comp_dict = czidoc.metadata


# try to write XML to file
xmlfile = czimd.writexml(filename)

# get the selected metadata as an object
mdata_sel = czimd.CziMetadata(filename)

# get the complete metadata as an object
mdata_comp = czimd.CziMetadataComplete(filename)

# convert to dictionary
mdata_sel_dict = czimd.obj2dict(mdata_sel)

# return a 7d array with dimension order STCZYXA
mdarray, dimstring = pylibczirw_tools.read_mdarray(filename)

# remove A dimension do display the array inside Napari
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(dimstring)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, mdarray, mdata_sel,
                           dim_order=dim_order,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
