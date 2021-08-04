from __future__ import annotations
from czitools import czi_metadata as czimd
from czitools import czi_read as czird
import napari
from aicspylibczi import CziFile
from utils import misc, napari_tools
import numpy as np

# adapt to your needs
defaultdir = r"D:\ImageData\CZI_Testfiles"

# open s simple dialog to select a CZI file
filename = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filename)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filename, dim2none=True)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# get the metadata as a dictionary
mdict = czimd.create_mdict_complete(filename, sort=False)

# get info from a specific scene
aicsczi = CziFile(filename)
scene = czimd.CziScene(aicsczi, 0)
print("Scene XY-Width-Height :", scene.xstart, scene.ystart, scene.width, scene.height)
print("Scene DimString :", scene.single_scene_dimstr)
print("Scene Shape :", scene.shape_single_scene)

# read pixel data
scene = aicsczi.read_mosaic(scale_factor=0.8, C=0)
# add S dimension
scene = scene[np.newaxis]

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, scene, mdata,
                           blending='additive',
                           calc_contrast=False,
                           auto_contrast=True,
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()