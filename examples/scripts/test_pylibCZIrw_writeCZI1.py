from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from utils import pylibczirw_tools
from utils import misc
import os
from tqdm.contrib.itertools import product

# define the CZI file to be read
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
filename = r"D:\Testdata_Zeiss\SPL66UK(a)-small.czi"

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# read 6D array from the CZI file
array6d = pylibczirw_tools.read_6darray(filename)

# write CZI from NumPy array
folder = r"d:\Temp\test_czi_write_withmd"
cziname = 'test_withmd.czi'
savename = os.path.join(folder, cziname)


# check dimensions and set to 1 if None
sizeS = misc.check_dimsize(mdata.dims.SizeS)
sizeT = misc.check_dimsize(mdata.dims.SizeT)
sizeZ = misc.check_dimsize(mdata.dims.SizeZ)
sizeC = misc.check_dimsize(mdata.dims.SizeC)


# open CZI document and write 2D planes
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
                     location=(0, 0)
                     )

    # write metadata explicitly
    czidoc.write_metadata()

    # write channel name
    for c in range(sizeC):
        print("Set Name for Channel: ", c, "to", "CH" + str(c+1))
        czidoc.write_metadata(channel_names={c: "CH" + str(c+1)})

print("Done")

# try to read the created CZI again# read 6D array from the CZI file
test = pylibczirw_tools.read_7darray(savename)