from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import napari
from utils import pylibczirw_tools
from utils import misc, napari_tools
import numpy as np

#filename = r'D:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
filename = r'C:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\tobacco_z=10_tiles.czi"

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# return a 7d array with dimension order STZCYXA
mdarray = pylibczirw_tools.read_7darray(filename)

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