# -*- coding: utf-8 -*-

#################################################################
# File        : czi_read.py
# Version     : 0.0.1
# Author      : sebi06
# Date        : 19.07.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from aicspylibczi import CziFile
from tqdm.contrib.itertools import product
import numpy as np
from czifiletools import czi_metadata as czimd
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from nptyping import Int, UInt, Float


def readczi(filename: str) -> np.ndarray:

    # get the CZI metadata
    md = czimd.CziMetadata(filename)

    # read CZI using aicspylibczi
    aicsczi = CziFile(filename)

    if not aicsczi.is_mosaic():

        # get the shape for the 1st scene
        scene = czimd.CziScene(aicsczi, sceneindex=0)
        shape_all = scene.shape_single_scene

        # only update the shape for the scene if the CZI has an S-Dimension
        if scene.hasS:
            shape_all[scene.posS] = md.dims.SizeS

        print('Shape all Scenes : ', shape_all)
        print('DimString all Scenes : ', scene.single_scene_dimstr)

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
                idl_scene[aicsczi.dims.index('S')] = 0
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
                idl_all[aicsczi.dims.index('S')] = s

                # cast current stack into the stack for all scenes
                all_scenes[tuple(idl_all)] = current_scene[tuple(idl_scene)]

            # if there is no S-Dimension use the stack directly
            if not scene.hasS:
                all_scenes = current_scene

        print('Shape all (no mosaic)', all_scenes.shape)

    if aicsczi.is_mosaic():

        # get data for 1st scene and create the required shape for all scenes
        scene = czimd.CziScene(aicsczi, sceneindex=0)
        shape_all = scene.shape_single_scene
        shape_all[scene.posS] = md.dims.SizeS
        print('Shape all Scenes : ', shape_all)
        print('DimString all Scenes : ', scene.single_scene_dimstr)

        # create empty array to hold all scenes
        all_scenes = np.empty(shape_all, dtype=md.npdtype)

        # loop over scenes if CZI is not Mosaic
        for s in range(md.dims.SizeS):
            scene = czimd.CziScene(aicsczi, sceneindex=s)

            # create a slice object for all_scenes array
            if not scene.isRGB:
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            if scene.isRGB:
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 3)

            # update the entry with the current S index
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
                                                           scale_factor=1.0,
                                                           H=h,
                                                           T=t,
                                                           Z=z,
                                                           C=c)

                    print('Shape Single Scene : ', scene_array_htzc.shape)
                    print('Min-Max Single Scene : ', np.min(scene_array_htzc), np.max(scene_array_htzc))

                    # create slide object for the current mosaic scene
                    # idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                    idl_all[scene.posS] = s
                    idl_all[scene.posH] = h
                    idl_all[scene.posT] = t
                    idl_all[scene.posZ] = z
                    idl_all[scene.posC] = c

                    # cast the current scene into the stack for all scenes
                    all_scenes[tuple(idl_all)] = scene_array_htzc

            if scene.hasT is False and scene.hasZ is False:

                # create an array for the scene
                for c in range(scene.sizeC):
                    scene_array_c = aicsczi.read_mosaic(region=(scene.xstart,
                                                                scene.ystart,
                                                                scene.width,
                                                                scene.height),
                                                        scale_factor=1.0,
                                                        C=c)

                    print('Shape Single Scene : ', scene_array_c.shape)

                    idl_all[scene.posS] = s
                    idl_all[scene.posC] = c

                    # cast the current scene into the stack for all scenes
                    all_scenes[tuple(idl_all)] = scene_array_c

    return all_scenes