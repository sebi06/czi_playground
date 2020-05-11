# -*- coding: utf-8 -*-

#################################################################
# File        : ometifftools.py
# Version     : 0.1
# Author      : czsrh
# Date        : 20.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import tifffile
import os
import numpy as np
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
import imgfileutils as imf

try:
    import javabridge as jv
    import bioformats
except ImportError as error:
    # Output expected ImportErrors.
    print(error.__class__.__name__ + ": " + error.msg)


def write_ometiff(filepath, img,
                  scalex=0.1,
                  scaley=0.1,
                  scalez=1.0,
                  dimorder='TZCYX',
                  pixeltype=np.uint16,
                  swapxyaxes=True,
                  series=1):
    """
    This function will write an OME-TIFF file to disk.
    The out 6D array has the following dimension order:

    [T, Z, C, Y, X] if swapxyaxes = True

    [T, Z, C, X, Y] if swapxyaxes = False
    """

    # Dimension STZCXY
    if swapxyaxes:
        # swap xy to write the OME-Stack with the correct shape
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[4]
        SizeY = img.shape[3]

    if not swapxyaxes:
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
        SizeC = img.shape[2]
        SizeX = img.shape[3]
        SizeY = img.shape[4]

    # Getting metadata info
    omexml = bioformats.omexml.OMEXML()
    omexml.image(series - 1).Name = filepath

    for s in range(series):
        p = omexml.image(s).Pixels
        p.ID = str(s)
        p.SizeX = SizeX
        p.SizeY = SizeY
        p.SizeC = SizeC
        p.SizeT = SizeT
        p.SizeZ = SizeZ
        p.PhysicalSizeX = np.float(scalex)
        p.PhysicalSizeY = np.float(scaley)
        p.PhysicalSizeZ = np.float(scalez)
        if pixeltype == np.uint8:
            p.PixelType = 'uint8'
        if pixeltype == np.uint16:
            p.PixelType = 'uint16'
        p.channel_count = SizeC
        p.plane_count = SizeZ * SizeT * SizeC
        p = writeOMETIFFplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder)

        for c in range(SizeC):
            # if pixeltype == 'unit8':
            if pixeltype == np.uint8:
                p.Channel(c).SamplesPerPixel = 1

            if pixeltype == np.uint16:
                p.Channel(c).SamplesPerPixel = 2

        omexml.structured_annotations.add_original_metadata(bioformats.omexml.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml(encoding='utf-8')

    # write file and save OME-XML as description
    tifffile.imwrite(filepath, img, metadata={'axes': dimorder}, description=xml)

    return filepath


def writeOMETIFFplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCXY', verbose=False):

    if order == 'TZCYX' or order == 'TZCXY':

        pixel.DimensionOrder = bioformats.omexml.DO_XYCZT
        counter = 0
        for t in range(SizeT):
            for z in range(SizeZ):
                for c in range(SizeC):

                    if verbose:
                        print('Write PlaneTable: ', t, z, c),
                        sys.stdout.flush()

                    pixel.Plane(counter).TheT = t
                    pixel.Plane(counter).TheZ = z
                    pixel.Plane(counter).TheC = c
                    counter = counter + 1

    return pixel


def write_ometiff_aicsimageio(savepath, imgarray, metadata,
                              reader='aicsimageio',
                              overwrite=False):
    """Write an OME-TIFF file from an image array based on the metadata

    :param filepath: savepath of the OME-TIFF stack 
    :type filepath: str
    :param imgarray: multi-dimensional image array
    :type imgarray: NumPy.Array
    :param metadata: metadata dictionary with the required information
    to create an correct OME-TIFF file
    :type metadata: dict
    :param reader: string (aicsimagio or czifile) specifying
    the used reader, defaults to aicsimageio
    :type metadata: str
    :param overwrite: option to overwrite an existing OME-TIFF, defaults to False
    :type overwrite: bool, optional
    """

    # define scaling from metadata or use defualt scaling
    try:
        pixels_physical_size = [metadata['XScale'],
                                metadata['YScale'],
                                metadata['ZScale']]
    except KeyError as e:
        print('Key not found:', e)
        print('Use default scaling XYZ=1')
        pixels_physical_size = [1, 1, 1]

    # define channel names list from metadata
    try:
        channel_names = []
        for ch in metadata['Channels']:
            channel_names.append(ch)
    except KeyError as e:
        print('Key not found:', e)
        channel_names = None

    # get the dimensions and their position inside the dimension string
    if reader == 'aicsimageio':

        dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(metadata['Axes_aics'])

        # if the array has more than 5 dimensions then remove the S dimension
        # because it is not supported by OME-TIFF
        if len(imgarray.shape) > 5:
            try:
                imgarray = np.squeeze(imgarray, axis=dims_dict['S'])
            except Exception:
                print('Could not remover S Dimension from string.)')

        # remove the S character from the dimension string
        new_dimorder = metadata['Axes_aics'].replace('S', '')

    if reader == 'czifile':

        new_dimorder = metadata['Axes']
        dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(metadata['Axes'])
        """
        '0': 'Sample',  # e.g. RGBA
        'X': 'Width',
        'Y': 'Height',
        'C': 'Channel',
        'Z': 'Slice',  # depth
        'T': 'Time',
        'R': 'Rotation',
        'S': 'Scene',  # contiguous regions of interest in a mosaic image
        'I': 'Illumination',  # direction
        'B': 'Block',  # acquisition
        'M': 'Mosaic',  # index of tile for compositing a scene
        'H': 'Phase',  # e.g. Airy detector fibers
        'V': 'View',  # e.g. for SPIM
        """

        to_remove = []

        # list of unspupported dims for writing an OME-TIFF
        dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', '0']

        for dim in dims:
            if dims_dict[dim] >= 0:
                # remove the CZI DIMENSION character from the dimension string
                new_dimorder = new_dimorder.replace(dim, '')
                # add dimension index to the list of axis to be removed
                to_remove.append(dims_dict[dim])
                print('Remove Dimension:', dim)

        # create tuple with dimensions to be removed
        dims2remove = tuple(to_remove)
        # remove dimensions from array
        imgarray = np.squeeze(imgarray, axis=dims2remove)

    """
    # write the array as an OME-TIFF incl. the metadata
    try:
        with ome_tiff_writer.OmeTiffWriter(savepath, overwrite_file=overwrite) as writer:
            writer.save(imgarray,
                        channel_names=channel_names,
                        image_name=os.path.basename((savepath)),
                        pixels_physical_size=pixels_physical_size,
                        channel_colors=None,
                        dimension_order=new_dimorder)
            writer.close()

        print('Saved image as: ', savepath)
    except Exception:
        print('Could not write OME-TIFF')
        savepath = None
    """

    with ome_tiff_writer.OmeTiffWriter(savepath, overwrite_file=overwrite) as writer:
        writer.save(imgarray,
                    channel_names=channel_names,
                    image_name=os.path.basename((savepath)),
                    pixels_physical_size=pixels_physical_size,
                    channel_colors=None,
                    dimension_order=new_dimorder)
        writer.close()
        print('Saved image as: ', savepath)

    return savepath


def correct_omeheader(omefile,
                      old=("2012-03", "2013-06", r"ome/2016-06"),
                      new=("2016-06", "2016-06", r"OME/2016-06")
                      ):

    tif = tifffile.TiffFile(omefile)
    array = tif.asarray()
    omexml_string = tif.ome_metadata

    for ostr, nstr in zip(old, new):
        print('Replace: ', ostr, 'with', nstr)
        omexml_string = omexml_string.replace(ostr, nstr)

    tifffile.imsave(omefile, array,
                    photometric='minisblack',
                    description=omexml_string)

    tif.close()

    print('Updated OME Header.')
