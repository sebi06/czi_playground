from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import napari
from czitools import pylibczirw_tools
from czitools import misc, napari_tools
import numpy as np

filename = r"/datadisk1/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=3_Z=4_CH=2.czi"

with pyczi.open_czi(filename) as czidoc:
    mdata = pyczi.CziMetadata(czidoc.raw_metadata)

    print(mdata.image.__dict__)
    print(mdata.scale.__dict__)
    print(mdata.channelinfo.__dict__)
    print(mdata.microscope.__dict__)
    print(mdata.objective.__dict__)
    print(mdata.detector.__dict__)
    print(mdata.wells.__dict__)
    print(mdata.info.__dict__)

