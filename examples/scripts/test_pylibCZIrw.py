# -*- coding: utf-8 -*-

#################################################################
# File        : test_pylibczirw.py
# Version     : 0.0.1
# Author      : sebi06
# Date        : 11.08.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
from utils import misc
import os

# open s simple dialog to select a CZI file
filename = os.path.abspath("../../testdata/w96_A1+A2.czi")

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# open the CZI document to read the
with pyczi.open_czi(filename) as czidoc:

    # show all dimensions
    total_bbox = czidoc.total_bounding_box
    for k, v in total_bbox.items():
        print(k, v)

    # get information about the scenes etc.
    sc_bbox = czidoc.scenes_bounding_rectangle
    total_rect = czidoc.total_bounding_rectangle
    pixeltype_ch = czidoc.get_channel_pixel_type(0)
    pixeltypes = czidoc.pixel_types
    print('Real Pixeltypes in CZI file : ', pixeltypes)

    # read a simple 2d image plane
    roi = (300, 300, 300, 600)
    image2d = czidoc.read(plane={'C': 0}, scene=0, zoom=1.0)
    #image2d = czidoc.read(plane={'C': 0}, scene=0, pixel_type="Gray8")
    #image2d = czidoc.read(plane={'C': 0}, scene=0, roi=roi, pixel_type="Gray8")
    print(image2d.shape)
    print('Pixeltype after conversion during reading : ', image2d.dtype)

# Create two subplots and unpack the output array immediately
f, ax1 = plt.subplots(1, 1)
ax1.imshow(image2d, interpolation='nearest', cmap='Reds_r')
plt.show()

# store metadata inside Pandas dataframe
mdf = misc.md2dataframe(mdata)
print(mdf)
