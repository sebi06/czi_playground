from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import napari
from utils import pylibczirw_tools
from utils import misc, napari_tools

filename = r'D:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

sceneid = 0

single_scene = czimd.CziScene(filename, sceneid)

print('Done')