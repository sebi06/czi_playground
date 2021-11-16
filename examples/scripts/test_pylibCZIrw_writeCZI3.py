#from czitools import pylibczirw_metadata as czimd
from czitools import czi_metadata as czimd
from pylibCZIrw import czi as pyczi
from aicspylibczi import CziFile
from utils import pylibczirw_tools
from utils import misc
import os
from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd
import napari
from utils import processing_tools as pt


# define the CZI file to be read
filename = r"D:\Testdata_Zeiss\SPL66UK(a)-small.czi"

# write CZI from NumPy array
folder = r"d:\Temp\test_czi_write"
cziname = 'test_tilewrite.czi'
savename = os.path.join(folder, cziname)


# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

tiles = pd.DataFrame(columns = ["m", "z", "height", "width", "x", "y"])

c = -1
for tileinfo, bbox in mdata.bbox.all_tiles.items():
    tiles.loc[len(tiles.index)] = [tileinfo.m_index, tileinfo.dimension_coordinates['Z'], bbox.h, bbox.w, bbox.x, bbox.y]

# sort the dataframe by the m index
tiles = tiles.sort_values(by=['m'])

# read CZI using aicspylibczi
czi = CziFile(filename)

# check dimensions and set to 1 if None
sizeS = misc.check_dimsize(mdata.dims.SizeS)
sizeT = misc.check_dimsize(mdata.dims.SizeT)
sizeZ = misc.check_dimsize(mdata.dims.SizeZ)
sizeC = misc.check_dimsize(mdata.dims.SizeC)

# create empty 2D image and define the dimension order to be STZCYX
#array6d = np.empty([sizeS,
#                    sizeT,
#                    1,
#                    sizeC,
#                    mdata.dims.SizeY,
#                    mdata.dims.SizeX,
#                    3 if mdata.isRGB else 1], dtype=mdata.npdtype)


# loop over all individual tile stacks
tile_rois = {}

# open CZI document for writing and write 2D planes
with pyczi.create_czi(savename) as czidoc:

    for s, t, c in product(range(sizeS),
                           range(sizeT),
                           range(sizeC)):

        for m in range(mdata.dims.SizeM):

            # read the z-stacl per tile
            tile_zstack, shp = czi.read_image(M=m)

            # process the tile stack here --> find z-plane
            plane2d, fv = pt.get_sharpest_plane(np.squeeze(tile_zstack))

            #zstack, new_shp = czi.read_image(M=m, Z=zplane)

            # get the xy position
            # selecting the correct tile from tile dataframe  based on conditions
            tile = tiles[(tiles['m'] == m) & (tiles['z'] == 0)]
            xpos = tile.x.values[0]
            ypos = tile.y.values[0]
            width = tile.width.values[0]
            height = tile.height.values[0]

            # insert the new stack into the predefined array6d
            #array6d[s, t, 0, c, ypos:(ypos+height), xpos:(xpos+width), 0] = plane2d

            # write new czi
            # write the 2D plane to the correct position
            print("Write new Tile to XY:", tile.x.values[0], tile.y.values[0])

            # remove all dimensions of size = 1
            plane2d = np.squeeze(plane2d)
            # add pixel type at the end
            plane2d = plane2d[..., np.newaxis]

            czidoc.write(plane2d, plane={"T": t,
                                         "Z": 0,
                                         "C": c},
                         scene=s,
                         location=(tile.x.values[0], tile.y.values[0])
                         )
    # write metadata explicitly
    czidoc.write_metadata()

# read 6D array from the CZI file
test = pylibczirw_tools.read_7darray(savename)

viewer = napari.view_image(test, colormap='gray')
napari.run()



# write CZI from NumPy array
folder = r"d:\Temp\test_czi_write"
cziname = 'test_multiscene.czi'
savename = os.path.join(folder, cziname)

# check dimensions and set to 1 if None
sizeS = misc.check_dimsize(mdata.dims.SizeS)
sizeT = misc.check_dimsize(mdata.dims.SizeT)
sizeZ = misc.check_dimsize(mdata.dims.SizeZ)
sizeC = misc.check_dimsize(mdata.dims.SizeC)

# open CZI document for writing and write 2D planes
with pyczi.create_czi(savename) as czidoc:

    # write a 2D plane into the CZI
    for s, t, z, c in product(range(sizeS),
                              range(sizeT),
                              range(sizeZ),
                              range(sizeC)):

        # get single 2D plane
        plane2d = array6d[s, t, z, c]

        # write the 2D plane to the correct position
        czidoc.write(plane2d, plane={"T": t,
                                     "Z": z,
                                     "C": c},
                              scene=s,
                              location=mdata.bbox.all_scenes[s]
                     )

    # write channel name
    for c in range(sizeC):
        czidoc.write_metadata(channel_names={c: "CH_" + str(c+1)})

print("Done")

# try to read the created CZI again# read 6D array from the CZI file
test = pylibczirw_tools.read_7darray(savename)
