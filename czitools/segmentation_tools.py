# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_tools.py
# Version     : 1.0
# Author      : sebi06
# Date        : 22.03.2021
#
# Kudos also to: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/docs/demo.ipynb
#
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################


import sys
from time import process_time, perf_counter
import os
from glob import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from aicsimageio import AICSImage, imread
from skimage import io, measure, segmentation
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, black_tophat, disk, square
from skimage.morphology import ball, closing, square, dilation
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import binary_opening
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_triangle, rank, median, gaussian, sobel
from skimage.segmentation import clear_border, watershed, random_walker, relabel_sequential
from skimage.color import label2rgb
from skimage.util import invert
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union


class ObjParams(NamedTuple):
    name: str
    median: Optional[float]
    mean: Optional[float]


def get_params_median_mean(labels: np.ndarray,
                           parameter: str = "feret_diameter_max") -> Optional[ObjParams]:

    # make sure this is a labeld image
    labels = label(labels)

    # get the region properties
    props = regionprops(labels)
    values = []

    try:
        # get the individual values for each object and calculate the median
        for prop in props:
            values.append(prop[parameter])

        values_array = np.asarray(values)

        objparams = ObjParams(parameter,
                              np.median(values_array),
                              values_array.mean())

    except Exception as e:
        print("Parameter:", parameter, " does not exist.")
        objparams = ObjParams(parameter,
                              None,
                              None)

    return objparams


def apply_watershed(binary: np.ndarray,
                    estimate_min_distance: bool = False,
                    min_distance: int = 10):
    """Apply normal watershed to a binary image

    :param binary: binary images from segmentation
    :type binary: NumPy.Array
    :param estimate_min_distance: try to estimate the minimum distance.
    Overrides min_dinstance, defaults to False
    :type estimate_min_distance: bool, optional
    :param min_distance: minimum peak distance [pixel], defaults to 10
    :type min_distance: int, optional
    :return: mask - mask with separeted objects
    :rtype: NumPy.Array
    """

    # create real binary image
    binary = binary > 0

    # create distance map
    distance = distance_transform_edt(binary)

    if estimate_min_distance:
        feretmax = get_params_median_mean(binary,
                                          parameter="feret_diameter_max")

        min_distance = int(np.round(feretmax.median / 2 * 0.9, 0))
        print("Estimated Minimum Distance for Watershed:", min_distance)

    # create the seeds
    peak_idx = peak_local_max(distance,
                              labels=binary,
                              min_distance=min_distance,
                              # num_peaks_per_label=1,
                              )

    # create peak mask
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True

    # label maxima
    markers, num_features = ndimage.label(peak_mask)

    # apply watershed
    mask = watershed(-distance, markers,
                     mask=binary,
                     watershed_line=True).astype(np.int)

    return mask


def apply_watershed_adv(image2d: np.ndarray,
                        segmented: np.ndarray,
                        filtermethod_ws: str = "median",
                        filtersize_ws: int = 3,
                        estimate_min_distance: bool = False,
                        min_distance: int = 10,
                        radius: int = 1):
    """Apply advanced watershed to a binary image

    :param image2d: 2D image with pixel intensities
    :type image2d: NumPy.Array
    :param segmented: binary images from initial segmentation
    :type segmented: NumPy.Array
    :param filtermethod_ws: choice of filter method, defaults to 'median'
    :type filtermethod_ws: str, optional
    :param filtersize_ws: size paramater for the selected filter, defaults to 3
    :type filtersize_ws: int, optional
    :param estimate_min_distance: try to estimate the minimum distance.
    Overrides min_dinstance, defaults to False
    :type estimate_min_distance: bool, optional
    :param min_distance: minimum peak distance [pixel], defaults to 2
    :type min_distance: int, optional
    :param radius: radius for dilation disk, defaults to 1
    :type radius: int, optional
    :return: mask - binary mask with separated objects
    :rtype: NumPy.Array
    """

    # convert to float
    image2d = image2d.astype(np.float)

    # rescale 0-1
    image2d = rescale_intensity(image2d, in_range='image', out_range=(0, 1))

    # filter image
    if filtermethod_ws == 'median':
        image2d = median(image2d, disk(filtersize_ws))
    if filtermethod_ws == 'gauss':
        image2d = gaussian(image2d, sigma=filtersize_ws, mode='reflect')

    # create the seeds
    # if image2d.dtype == np.float64:
    #    image2d = image2d.astype(np.float32)
    labels = label(segmented).astype(np.int32)

    if estimate_min_distance:
        feretmax = get_params_median_mean(segmented,
                                          parameter="feret_diameter_max")

        min_distance = int(np.round(feretmax.median / 2 * 0.9, 0))
        print("Estimated Minimum Distance for Watershed:", min_distance)

    peak_idx = peak_local_max(image2d,
                              labels=labels,
                              min_distance=min_distance,
                              # num_peaks_per_label=1,
                              # indices=False
                              )

    # create peak mask
    peak_mask = np.zeros_like(image2d, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True

    # create the seeds
    seed = dilation(peak_mask, selem=disk(radius))

    # create watershed map
    watershed_map = -1 * distance_transform_edt(segmented)

    # create mask
    mask = watershed(watershed_map,
                     markers=label(seed),
                     mask=segmented,
                     watershed_line=True).astype(np.int)

    return mask


def segment_threshold(image2d,
                      filtermethod='median',
                      filtersize=3,
                      threshold='triangle',
                      split_ws=True,
                      min_distance=30,
                      ws_method='ws_adv',
                      radius=1,
                      dtypemask=np.int16):
    """Segment an image using the following steps:
    - filter image
    - threshold image
    - apply watershed

    :param image2d: 2D image with pixel intensities
    :type image2d: NumPy.Array
    :param filtermethod: choice of filter method, defaults to 'median'
    :type filtermethod: str, optional
    :param filtersize: size paramater for the selected filter, defaults to 3
    :type filtersize: int, optional
    :param threshold: choice of thresholding method, defaults to 'triangle'
    :type threshold: str, optional
    :param split_ws: enable splitting using watershed, defaults to True
    :type split_ws: bool, optional
    :param min_distance: minimum peak distance [pixel], defaults to 30
    :type min_distance: int, optional
    :param ws_method: choice of watershed method, defaults to 'ws_adv'
    :type ws_method: str, optional
    :param radius: radius for dilation disk, defaults to 1
    :type radius: int, optional
    :param dtypemask: datatype of output mask, defaults to np.int16
    :type dtypemask: np.dtype, optional
    :return: mask - binary mask
    :rtype: NumPy.Array
    """

    # filter image
    if filtermethod is None:
        image2d_filtered = image2d
    if filtermethod == 'median':
        image2d_filtered = median(image2d, selem=disk(filtersize))
    if filtermethod == 'gauss':
        image2d_filtered = gaussian(image2d, sigma=filtersize, mode='reflect')

    # threshold image and run marker-based watershed
    binary = autoThresholding(image2d_filtered, method=threshold)

    # apply watershed
    if split_ws:

        if ws_method == 'ws':
            mask = apply_watershed(binary,
                                   min_distance=min_distance)

        if ws_method == 'ws_adv':
            mask = apply_watershed_adv(image2d, binary,
                                       min_distance=min_distance,
                                       radius=radius)

    if not split_ws:
        # label the objects
        mask, num_features = ndimage.label(binary)
        mask = mask.astype(np.int)

    return mask.astype(dtypemask)


def autoThresholding(image2d,
                     method='triangle',
                     radius=10,
                     value=50):
    """Autothreshold an 2D intensity image which is calculated using:
    binary = image2d >= thresh

    :param image2d: input image for thresholding
    :type image2d: NumPy.Array
    :param method: choice of thresholding method, defaults to 'triangle'
    :type method: str, optional
    :param radius: radius of disk when using local Otsu threshold, defaults to 10
    :type radius: int, optional
    :param value: manual threshold value, defaults to 50
    :type value: int, optional
    :return: binary - binary mask from thresholding
    :rtype: NumPy.Array
    """

    # calculate global Otsu threshold
    if method == 'global_otsu':
        thresh = threshold_otsu(image2d)

    # calculate local Otsu threshold
    if method == 'local_otsu':
        thresh = rank.otsu(image2d, disk(radius))

    if method == 'value_based':
        thresh = value

    if method == 'triangle':
        thresh = threshold_triangle(image2d)

    binary = image2d >= thresh

    return binary


def subtract_background(image,
                        elem='disk',
                        radius=50,
                        light_bg=False):
    """Background substraction using structure element.
    Slightly adapted from: https://forum.image.sc/t/background-subtraction-in-scikit-image/39118/4

    :param image: input image
    :type image: NumPy.Array
    :param elem: type of the structure element, defaults to 'disk'
    :type elem: str, optional
    :param radius: size of structure element [pixel], defaults to 50
    :type radius: int, optional
    :param light_bg: light background, defaults to False
    :type light_bg: bool, optional
    :return: image with background subtracted
    :rtype: NumPy.Array
    """
    # use 'ball' here to get a slightly smoother result at the cost of increased computing time
    if elem == 'disk':
        str_el = disk(radius)
    if elem == 'ball':
        str_el = ball(radius)

    if light_bg:
        img_subtracted = black_tophat(image, str_el)
    if not light_bg:
        img_subtracted = white_tophat(image, str_el)

    return img_subtracted


def sobel_3d(image):
    kernel = np.asarray([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], [
            [0, 1, 0],
            [1, -6, 1],
            [0, 1, 0]
        ], [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
    ])
    return ndimage.convolve(image, kernel)


def split_touching_objects(binary: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    """
    Takes a binary image and draws cuts in the objects similar to the ImageJ watershed algorithm.
    See also
    --------
    .. [0] https://imagej.nih.gov/ij/docs/menus/process.html#watershed
    """
    # binary = np.asarray(binary)

    # typical way of using scikit-image watershed
    distance = ndimage.distance_transform_edt(binary)
    blurred_distance = gaussian(distance, sigma=sigma)
    fp = np.ones((3,) * binary.ndim)
    coords = peak_local_max(blurred_distance, footprint=fp, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    labels = watershed(-blurred_distance, markers, mask=binary)

    # identify label-cutting edges
    if len(binary.shape) == 2:
        edges = sobel(labels)
        edges2 = sobel(binary)
    else:  # assuming 3D
        edges = sobel_3d(labels)
        edges2 = sobel_3d(binary)

    almost = np.logical_not(np.logical_xor(edges != 0, edges2 != 0)) * binary
    return binary_opening(almost)


def erode_labels(segmentation, erosion_iterations, relabel=True):

    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = list()
    regions = regionprops(segmentation)

    def erode_mask(segmentation_labels, label_id, erosion_iterations, relabel=True):
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = ndimage.binary_erosion(only_current_label_id, iterations=erosion_iterations)

        if relabel:
            # relabeled_eroded = np.where(eroded == 1, label_id, 0)
            return (np.where(eroded == 1, label_id, 0))

        if not relabel:
            return (eroded)

    for i in range(len(regions)):
        label_id = regions[i].label
        list_of_eroded_masks.append(erode_mask(segmentation,
                                               label_id,
                                               erosion_iterations,
                                               relabel=relabel))

    # convert list of numpy arrays to stacked numpy array
    final_array = np.stack(list_of_eroded_masks)

    # max_IP to reduce the stack of arrays, each containing one labelled region, to a single 2D np array.
    final_array_labelled = np.sum(final_array, axis=0)

    return final_array_labelled


def area_filter(im: np.ndarray, area_min: int = 10, area_max: int = 100000) -> np.ndarray:
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
