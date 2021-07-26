from aicspylibczi import CziFile
import xmltodict
from lxml import etree as ET
import imgfile_tools as imf
import numpy as np


filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Smart_Microscopy_Workshop\datasets\OverViewScan.czi"
metadata = {}

# get metadata dictionary using aicspylibczi
czi_aicspylibczi = CziFile(filename)
metadatadict_czi = xmltodict.parse(ET.tostring(czi_aicspylibczi.meta))

# Get the shape of the data, the coordinate pairs are (start index, size)
metadata['dims_aicspylibczi'] = czi_aicspylibczi.dims_shape()[0]
metadata['axes_aicspylibczi'] = czi_aicspylibczi.dims
metadata['size_aicspylibczi'] = czi_aicspylibczi.size
metadata['czi_isMosaic'] = czi_aicspylibczi.is_mosaic()
print('CZI is Mosaic :', metadata['czi_isMosaic'])

metadata['SizeS'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
metadata['SizeT'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeT'])
metadata['SizeZ'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
metadata['SizeC'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
metadata['SizeX'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeX'])
metadata['SizeY'] = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeY'])
