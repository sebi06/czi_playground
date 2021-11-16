# -*- coding: utf-8 -*-

#################################################################
# File        : czi_read.py
# Version     : 0.0.4
# Author      : sebi06
# Date        : 15.11.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from aicspylibczi import CziFile
from tqdm.contrib.itertools import product
import numpy as np
from czitools import czi_metadata as czimd
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def read(filename: str) -> Tuple[np.ndarray, czimd.CziMetadata]:
    """Read the pixel data of an CZI image file. Scaling factor
    of 1.0 will be used.

    :param filename: filename of the CZI to be read
    :return: CZI pixel data and the CziMetadata class
    """

    # get the CZI metadata
    md = czimd.CziMetadata(filename)

    # read CZI using aicspylibczi
    aicsczi = CziFile(filename)

    # in case of a non-mosaic CZI file
    if not aicsczi.is_mosaic():

        # get all scenes
        all_scenes, md = read_nonmosaic(filename=filename)

    # in case of a mosaic CZI file
    if aicsczi.is_mosaic():

        # get all scenes
        all_scenes, md = read_mosaic(filename=filename, scale=1.0)

    return all_scenes, md


def read_nonmosaic(filename: str) -> Tuple[Union[np.ndarray, None], czimd.CziMetadata]:
    """Read CZI pixel data from non-mosaic image data
    :param filename: filename of the CZI file to be read
    :return: CZI pixel data and the CziMetadata class
    """

    # get the CZI metadata
    md = czimd.CziMetadata(filename)

    # read CZI using aicspylibczi
    aicsczi = CziFile(filename)

    if aicsczi.is_mosaic():
        # check if this CZIn is really a non-mosaic file
        print("CZI is a mosaic file. Please use the readczi_mosaic method instead")
        return None, md

    # get the shape for the 1st scene
    scene = czimd.CziScene(aicsczi, sceneindex=0)
    shape_all = scene.shape_single_scene

    # only update the shape for the scene if the CZI has an S-Dimension
    if scene.hasS:
        shape_all[scene.posS] = md.dims.SizeS

    print("Shape all Scenes : ", shape_all)
    print("DimString all Scenes : ", scene.single_scene_dimstr)

    # create an empty array with the correct dimensions
    all_scenes = np.empty(aicsczi.size, dtype=md.npdtype)

    # loop over scenes if CZI is not a mosaic image
    if md.dims.SizeS is None:
        sizeS = 1
    else:
        sizeS = md.dims.SizeS

    for s in range(sizeS):

        # read the image stack for the current scene
        current_scene, shp = aicsczi.read_image(S=s)

        # create th index lists containing the slice objects
        if scene.hasS:
            idl_scene = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            idl_scene[aicsczi.dims.index("S")] = 0
            idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            idl_all[aicsczi.dims.index("S")] = s

            # cast current stack into the stack for all scenes
            all_scenes[tuple(idl_all)] = current_scene[tuple(idl_scene)]

        # if there is no S-Dimension use the stack directly
        if not scene.hasS:
            all_scenes = current_scene

    print("Shape all scenes (no mosaic)", all_scenes.shape)

    return all_scenes, md


def read_mosaic(filename: str, scale: float=1.0) -> Tuple[Union[np.ndarray, None], czimd.CziMetadata]:
    """Read the pixel data of an CZI image file with an option scale factor
    to read the image with lower resolution and array size

    :param filename: filename of the CZI mosaic file to be read
    :param scale: scaling factor when reading the mosaic.
    :return: CZI pixel data and the updated CziMetadata class
    """

    # do not allow scale > 1.0
    if scale > 1.0:
        print("Scale factor > 1.0 is not recommended. Using scale = 1.0.")
        scale = 1.0

    # get the CZI metadata
    md = czimd.CziMetadata(filename)

    # read CZI using aicspylibczi
    aicsczi = CziFile(filename)

    if not aicsczi.is_mosaic():
        # check if this CZI is really a non-mosaic file
        print("CZI is not a mosaic file. Please use the read_nonmosaic method instead")
        return None, md

    # get data for 1st scene and create the required shape for all scenes
    scene = czimd.CziScene(aicsczi, sceneindex=0)
    shape_all = scene.shape_single_scene

    if scene.hasS:
        shape_all[scene.posS] = md.dims.SizeS
    if not scene.hasS:
        num_scenes = 1

    print("Shape all Scenes (scale=1.0): ", shape_all)
    print("DimString all Scenes : ", scene.single_scene_dimstr)

    # create empty array to hold all scenes
    all_scenes = np.empty(shape_all, dtype=md.npdtype)

    resize_done = False

    # loop over scenes if CZI is not Mosaic
    for s in range(num_scenes):
        scene = czimd.CziScene(aicsczi, sceneindex=s)

        # create a slice object for all_scenes array
        if not scene.isRGB:
            #idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            idl_all = [slice(None, None, None)] * (len(shape_all) - 2)
        if scene.isRGB:
            #idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 3)
            idl_all = [slice(None, None, None)] * (len(shape_all) - 3)

        # update the entry with the current S index
        if not scene.hasS:
            idl_all[s] = s
        if scene.hasS:
            idl_all[scene.posS] = s

        # in case T-Z-H dimension are found
        if scene.hasT is True and scene.hasZ is True and scene.hasH is True:

            # read array for the scene
            for h, t, z, c in product(range(scene.sizeH),
                                      range(scene.sizeT),
                                      range(scene.sizeZ),
                                      range(scene.sizeC)):
                # read the array for the 1st scene using the ROI
                scene_array_htzc = aicsczi.read_mosaic(region=(scene.xstart,
                                                               scene.ystart,
                                                               scene.width,
                                                               scene.height),
                                                       scale_factor=scale,
                                                       H=h,
                                                       T=t,
                                                       Z=z,
                                                       C=c)

                print("Shape Single Scene (Scalefactor: ", scale, ": ", scene_array_htzc.shape)

                # check if all_scenes array must be resized due to scaling
                if scale < 1.0 and not resize_done:

                    shape_all[-1] = scene_array_htzc.shape[-1]
                    shape_all[-2] = scene_array_htzc.shape[-2]
                    all_scenes = np.resize(all_scenes, shape_all)

                    # add new entries to metadata
                    md = adaptmd_scale(md, scene_array_htzc.shape[-1], scene_array_htzc.shape[-2], scale=scale)
                    resize_done = True

                # create slide object for the current mosaic scene
                # idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                idl_all[scene.posS] = s
                idl_all[scene.posH] = h
                idl_all[scene.posT] = t
                idl_all[scene.posZ] = z
                idl_all[scene.posC] = c

                # cast the current scene into the stack for all scenes
                all_scenes[tuple(idl_all)] = scene_array_htzc

        # in case T-Z-H dimension are found
        if scene.hasT is True and scene.hasZ is True and scene.hasH is False:

            # read array for the scene
            for t, z, c in product(range(scene.sizeT),
                                   range(scene.sizeZ),
                                   range(scene.sizeC)):
                # read the array for the 1st scene using the ROI
                scene_array_tzc = aicsczi.read_mosaic(region=(scene.xstart,
                                                              scene.ystart,
                                                              scene.width,
                                                              scene.height),
                                                       scale_factor=scale,
                                                       T=t,
                                                       Z=z,
                                                       C=c)

                print("Shape Single Scene (Scalefactor: ", scale, ": ", scene_array_tzc.shape)

                # check if all_scenes array must be resized due to scaling
                if scale < 1.0 and not resize_done:
                    shape_all[-1] = scene_array_tzc.shape[-1]
                    shape_all[-2] = scene_array_tzc.shape[-2]
                    all_scenes = np.resize(all_scenes, shape_all)

                    # add new entries to metadata
                    md = adaptmd_scale(md, scene_array_tzc.shape[-1], scene_array_tzc.shape[-2], scale=scale)
                    resize_done = True

                # create slide object for the current mosaic scene
                # idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                idl_all[scene.posS] = s
                idl_all[scene.posT] = t
                idl_all[scene.posZ] = z
                idl_all[scene.posC] = c

                # cast the current scene into the stack for all scenes
                all_scenes[tuple(idl_all)] = scene_array_tzc



        if scene.hasT is False and scene.hasZ is False:

            # create an array for the scene
            for c in range(scene.sizeC):
                scene_array_c = aicsczi.read_mosaic(region=(scene.xstart,
                                                            scene.ystart,
                                                            scene.width,
                                                            scene.height),
                                                    scale_factor=scale,
                                                    C=c)

                print("Shape Single Scene (Scalefactor: ", scale, ": ", scene_array_c.shape)

                # check if all_scenes array must be resized due to scaling
                if scale < 1.0 and not resize_done:
                    #new_shape = shape_all
                    shape_all[-1] = scene_array_c.shape[-1]
                    shape_all[-2] = scene_array_c.shape[-2]
                    all_scenes = np.resize(all_scenes, shape_all)

                    # add new entries to metadata
                    md = adaptmd_scale(md, scene_array_c.shape[-1], scene_array_c.shape[-2], scale=scale)
                    resize_done = True

                idl_all[scene.posS] = s
                idl_all[scene.posC] = c

                # cast the current scene into the stack for all scenes
                all_scenes[tuple(idl_all)] = scene_array_c

    return all_scenes, md


def adaptmd_scale(metadata: czimd.CziMetadata,
                  newx: int,
                  newy: int,
                  scale: float=1.0) -> czimd.CziMetadata:
    """ Adapt some metadata due to reading the CZI using a scaling factor.

    :param metadata: CziMetadata class to be modified
    :param newx: new size of X dimension [pixel]
    :param newy: new size of Y dimension [pixel]
    :param scale: scale factor used to read the CZI image
    :return: updated CziMetadata class
    """

    # adapt the size entry
    setattr(metadata.dims, "SizeX_sf", newx)
    setattr(metadata.dims, "SizeY_sf", newy)

    # adapt the scaling and add scalefactor attribute
    setattr(metadata.scale, "scalefactorXY", scale)
    setattr(metadata.scale, "X_sf", np.round(metadata.scale.X / scale, 3))
    setattr(metadata.scale, "Y_sf", np.round(metadata.scale.Y / scale, 3))
    setattr(metadata.scale, "ratio_sf", {"xy": 1.0, "zx": metadata.scale.ratio["zx"] * scale})

    return metadata



