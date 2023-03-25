# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_cellpose.py
# Author      : sebi06
#
# CellPose: https://github.com/MouseLand/cellpose
# CellPose API: https://cellpose.readthedocs.io/en/latest/
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import numpy as np
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
from skimage import measure
import matplotlib.pyplot as plt
from skimage import measure, segmentation
from czitools import pylibczirw_metadata as czimd
from czitools import misc
from cellpose import models


def get_cpmodels_path(model_type: str,
                      modelbasedir: str,
                      use_gpu: bool = False) -> models.CellposeModel:
    """Get the cellpose models from a folder directly.

     Parameters
     ----------
     model_type : str
         Type of the CellPose model to be used
     modelbasedir : str
         Base directory where to look for the pretrained models
     use_gpu : bool, optional
         Use GPU-based model

     Returns
     -------
     models.CellposeModel
         CellPose model to be used for segmentation
     """

    # check if the provide model_type is actually valid
    if model_type not in ["2d_nuclei_fluo",
                          "2d_cyto",
                          "2d_cyto2",
                          "2d_cyto_fluo",
                          "2d_nuclei_fluo_test"]:

        print("No match for found for model_type:", model_type)
        return None
    else:
        # get the model locations
        if model_type == "2d_nuclei_fluo" or model_type == "2d_nuclei_fluo_test":
            print("Detected Modeltype:", model_type)
            models_path = [os.path.join(os.getcwd(), os.path.join(modelbasedir, "nucleitorch_0")),
                           os.path.join(os.getcwd(), os.path.join(
                               modelbasedir, "nucleitorch_1")),
                           os.path.join(os.getcwd(), os.path.join(
                               modelbasedir, "nucleitorch_2")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "nucleitorch_3"))]

        if model_type == "2d_cyto":
            print("Detected Modeltype:", model_type)
            models_path = [os.path.join(os.getcwd(), os.path.join(modelbasedir, "cytotorch_0")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cytotorch_1")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cytotorch_2")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cytotorch_3"))]

        if model_type == "2d_cyto2":
            print("Detected Modeltype:", model_type)
            models_path = [os.path.join(os.getcwd(), os.path.join(modelbasedir, "cyto2torch_0")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cyto2torch_1")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cyto2torch_2")),
                           os.path.join(os.getcwd(), os.path.join(modelbasedir, "cyto2torch_3"))]

        if model_type == "2d_cyto_fluo":
            print("Detected Modeltype:", model_type)
            models_path = [os.path.join(os.getcwd(), os.path.join(
                modelbasedir, "CP_cyto_fluorescent"))]

        print("Loading Cellpose Models from folder:", models_path)
        cp_model = models.CellposeModel(gpu=use_gpu, pretrained_model=models_path)

        return cp_model


def segment_objects_cellpose2d(img2d: np.ndarray,
                               cp_model: models.CellposeModel,
                               channels: List[int] = [0, 0],
                               rescale: bool = 1.0,
                               diameter: int = 17,
                               minsize_obj: int = 15,
                               tile: bool = True,
                               tile_overlap: float = 0.1,
                               cellprob_threshold: float = 0.0) -> np.ndarray:
    """Segment objects using CellPose2D models

    Parameters
    ----------
    img2d : np.ndarray
        2D image to be segmented
    cp_model : models.CellposeModel
        the CellPose2D model to be used for segmentation
    channels : List[int], optional
        Channel information for cellPose2D - Must be [0,0] in case of grayscale, by default [0, 0]
    rescale : bool, optional
        resize factor for each image, by default 1.0
    diameter : int, optional
        Estimated Diameter for the objects, by default 17
    minsize_obj : int, optional
        Minimum size for the objects [pixel], by default 15
    tile : bool, optional
        Tile image to ensure GPU/CPU memory usage limited (recommended), by default True
    tile_overlap : float, optional
        Fraction of overlap of tiles when computing flows, by default 0.1
    cellprob_threshold : float, optional
        all pixels with value above threshold kept for masks, decrease to find more and larger masks, by default 0.0

    Returns
    -------
    np.ndarray
        2D image with labels
    """

    # get the mask for a single image
    labels, _, _ = cp_model.eval([img2d],
                                 batch_size=8,
                                 channels=channels,
                                 diameter=diameter,
                                 min_size=minsize_obj,
                                 normalize=True,
                                 invert=False,
                                 rescale=rescale,
                                 do_3D=False,
                                 net_avg=True,
                                 tile=tile,
                                 tile_overlap=tile_overlap,
                                 augment=False,
                                 flow_threshold=0.4,
                                 cellprob_threshold=cellprob_threshold,
                                 progress=None)

    return labels[0]


# def load_cellpose_model(model_type='nuclei',
#                         gpu=True,
#                         net_avg=True):
#     """_summary_

#     Parameters
#     ----------
#     model_type : str, optional
#         _description_, by default 'nuclei'
#     gpu : bool, optional
#         _description_, by default True
#     net_avg : bool, optional
#         _description_, by default True

#     Returns
#     -------
#     _type_
#         _description_
#     """

#     # load cellpose model for cell nuclei using GPU or CPU
#     print('Loading Cellpose Model ...')

#     model = models.Cellpose(gpu=gpu,
#                             model_type=model_type,
#                             net_avg=net_avg)

    return model


def area_filter(im: np.ndarray, area_min: int = 10, area_max: int = 100000) -> Tuple[np.ndarray, int]:
    """
    Filters objects in an image based on their areas.

    Parameters
    ----------
    im : 2d-array, int
        Labeled segmentation mask to be filtered.
    area_min : int
        Minimum value for the area in units of square pixels.
    area_man : int
        Maximum value for the area in units of square pixels.

    Returns
    -------
    im_relab : 2d-array, int
        The relabeled, filtered image.
    num_labels : int
        The number of returned labels.
    """

    # Extract the region props of the objects.
    props = measure.regionprops(im)

    # Extract the areas and labels.
    areas = np.array([prop.area for prop in props])
    labels = np.array([prop.label for prop in props])

    # Make an empty image to add the approved cells.
    im_approved = np.zeros_like(im)

    # Threshold the objects based on area and eccentricity
    for i, _ in enumerate(areas):
        if areas[i] > area_min and areas[i] < area_max:
            im_approved += im == labels[i]

    # Relabel the image.
    im_filt, num_labels = measure.label(im_approved > 0, return_num=True)

    return im_filt, num_labels


def filter_labels(labels: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, int]:
    """Filter labels based on size.

    Args:
        labels (np.ndarray): Label Image
        min_size (int): Minimum size of labels [pixels]
        max_size (int): Maximum size of labels [pixels]

    Returns:
        Tuple[np.ndarray, int]: Label image with labels filtered by size and the number of labels left.
    """

    component_sizes = np.bincount(labels.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labels]
    labels[too_small_mask] = 0

    component_sizes = np.bincount(labels.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[labels]
    labels[too_big_mask] = 0

    # Relabel the image.
    labels, num_labels = measure.label(labels > 0, return_num=True)

    return labels, num_labels
