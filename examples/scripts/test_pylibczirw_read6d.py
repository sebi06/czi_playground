from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import napari
from utils import pylibczirw_tools
from utils import misc, napari_tools

#filename = r'D:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
filename = r'D:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

array6d = pylibczirw_tools.read_6darray(filename)

mdata.dimstring = "STZCYX"
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(mdata.dimstring)
setattr(mdata, "dim_order", dim_order)
setattr(mdata, "dim_index", dim_index)
setattr(mdata, "dim_valid", dim_valid)

# define the dimension string
print(array6d.shape)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, array6d, mdata,
                           blending="additive",
                           contrast='calc',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()