from czitools import czi_metadata as czimd
from czitools import czi_read as czird
import napari
from aicspylibczi import CziFile
from utils import misc, napari_tools

defaultdir = r"D:\Testdata_Zeiss\CZI_Testfiles"
filename = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filename)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filename, dim2none=True)
czi_scaling = czimd.CziScaling(filename, dim2none=True)
czi_channels = czimd.CziChannelInfo(filename)
czi_bbox = czimd.CziBoundingBox(filename)
czi_info = czimd.CziInfo(filename)
czi_objectives = czimd.CziObjectives(filename)
czi_detectors = czimd.CziDetector(filename)
czi_microscope =czimd.CziMicroscope(filename)
czi_sample = czimd.CziSampleInfo(filename)

# get the complete metadata at once
mdata = czimd.CziMetadata(filename)

# get the metadata as a dictionary
mdict = czimd.create_mdict_complete(filename)
for k,v in mdict.items():
    print(k, ' : ', v)

# get as pandas dataframe
df_md = misc.md2dataframe(mdict)
print(df_md[:10])



# write XML to disk
xmlfile = czimd.writexml_czi(filename)

# get the planetable for the CZI file
pt, csvfile = czimd.get_planetable(filename,
                          norm_time=True,
                          savetable=True,
                          separator=',',
                          index=True)

print(pt[:5])


# show array inside napari viewer
viewer = napari_tools.Viewer()

# specify the index of the Channel inside the array
aicsczi = CziFile(filename)
all_scenes = czird.readczi(filename)
scene = czimd.CziScene(aicsczi, 0)
#cpos = scene.single_scene_dimstr.find('C')

layers = napari_tools.show_napari(viewer, all_scenes, mdata,
                                  blending='additive',
                                  calc_contrast=False,
                                  auto_contrast=True,
                                  gamma=0.85,
                                  add_mdtable=True,
                                  rename_sliders=True)

napari.run()