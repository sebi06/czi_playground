# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_tools.py
# Author      : sebi06
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from aicsimageio import AICSImage, imread
from skimage import io, measure, segmentation
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, black_tophat, disk, square, ball, closing, square, dilation
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_triangle, rank, median, gaussian
from skimage.segmentation import clear_border, watershed, random_walker
from skimage.color import label2rgb
from skimage.util import invert
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from MightyMosaic import MightyMosaic
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union

# try:
#    print('Trying to find mxnet library ...')
#    import mxnet
# except (ImportError, ModuleNotFoundError) as error:
#    # Output expected ImportErrors.
#    print(error.__class__.__name__ + ": " + error.msg)
#    print('mxnet will not be used.')

try:
    print('Trying to find cellpose library ...')
    from cellpose import plot, transforms
    from cellpose import models, utils
except (ImportError, ModuleNotFoundError) as error:
    # Output expected ImportErrors.
    print(error.__class__.__name__ + ": " + error.msg)
    print('CellPose cannot be used.')

# try:
#     print('Trying to find tensorflow library ...')
#     # silence tensorflow output
#     from silence_tensorflow import silence_tensorflow
#     silence_tensorflow()
#     import tensorflow as tf
#     logging.getLogger("tensorflow").setLevel(logging.ERROR)
#     print('TensorFlow Version : ', tf.version.GIT_VERSION, tf.__version__)
# except (ImportError, ModuleNotFoundError) as error:
#     # Output expected ImportErrors.
#     print(error.__class__.__name__ + ": " + error.msg)
#     print('TensorFlow will not be used.')

# try:
#     print('Trying to find stardist library ...')
#     from stardist.models import StarDist2D
#     from csbdeep.utils import Path, normalize
# except (ImportError, ModuleNotFoundError) as error:
#     # Output expected ImportErrors.
#     print(error.__class__.__name__ + ": " + error.msg)
#     print('StarDist will not be used.')


class ObjParams(NamedTuple):
    name: str
    median: Optional[float]
    mean: Optional[float]

    """Just a little helper class"""


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
                    min_distance: int = 10) -> np.ndarray:
    """Apply normal watershed to a binary image

        :param binary: binary images from segmentation
        :type binary: NumPy.Array
        :param estimate_min_distance: try to estimate the minimum distance.
        Overrides min_distance, defaults to False
        :type estimate_min_distance: bool, optional
        :param min_distance: minimum peak distance [pixel], defaults to 10
        :type min_distance: int, optional
        :return: mask - mask with separated objects
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
                        radius: int = 1) -> np.ndarray:
    """Apply advanced watershed to a binary image

    :param image2d: 2D image with pixel intensities
    :type image2d: NumPy.Array
    :param segmented: binary images from initial segmentation
    :type segmented: NumPy.Array
    :param filtermethod_ws: choice of filter method, defaults to 'median'
    :type filtermethod_ws: str, optional
    :param filtersize_ws: size parameter for the selected filter, defaults to 3
    :type filtersize_ws: int, optional
    :param estimate_min_distance: try to estimate the minimum distance.
    Overrides min_distance, defaults to False
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


def segment_threshold(image2d: np.ndarray,
                      filtermethod='median',
                      filtersize=3,
                      threshold='triangle',
                      split_ws=True,
                      min_distance=30,
                      ws_method='ws_adv',
                      radius=1,
                      dtypemask=np.int16) -> np.ndarray:
    """Segment an image using the following steps:
    - filter image
    - threshold image
    - apply watershed

    :param image2d: 2D image with pixel intensities
    :type image2d: NumPy.Array
    :param filtermethod: choice of filter method, defaults to 'median'
    :type filtermethod: str, optional
    :param filtersize: size parameter for the selected filter, defaults to 3
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


def autoThresholding(image2d: np.ndarray,
                     method='triangle',
                     radius=10,
                     value=50) -> np.ndarray:
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
                        light_bg=False) -> np.ndarray:
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


def sobel_3d(image: np.ndarray) -> np.ndarray:
    """3D Sobel Filter

    Args:
        image (np.ndarray): Image to be filtered

    Returns:
        np.ndarray: Filtered image
    """

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


def erode_labels(segmentation: np.ndarray, erosion_iterations: int, relabel=True) -> np.ndarray:
    """Erode labels inside label image

    Args:
        segmentation (np.ndarray): Image with the label masks
        erosion_iterations (int): Number of iteration for the ersion
        relabel (bool, optional): Relabel the image . Defaults to True.

    Returns:
        np.ndarray: _description_
    """

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


def cutout_subimage(image2d,
                    startx=0,
                    starty=0,
                    width=100,
                    height=200):
    """Cutout a subimage ot of a bigger image

    :param image2d: the original image
    :type image2d: NumPy.Array
    :param startx: startx, defaults to 0
    :type startx: int, optional
    :param starty: starty, defaults to 0
    :type starty: int, optional
    :param width: width, defaults to 100
    :type width: int, optional
    :param height: height, defaults to 200
    :type height: int, optional
    :return: image2d - subimage cutted out from original image2d
    :rtype: NumPy.Array
    """

    image2d = image2d[starty:starty + height, startx:startx + width]

    return image2d


def segment_nuclei_cellpose2d(image2d, model,
                              channels=[0, 0],
                              rescale=None,
                              diameter=None,
                              verbose=False,
                              cellprob_threshold=0.5,
                              autotune=False):
    """Segment nucleus or cytosol using a cellpose model in 2D

    - define CHANNELS to run segmentation on
    - grayscale=0, R=1, G=2, B=3
    - channels = [cytoplasm, nucleus]
    - if NUCLEUS channel does not exist, set the second channel to 0
    - IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    - channels = [0,0] # IF YOU HAVE GRAYSCALE
    - channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    - channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus


    :param image2d: 2D image
    :type image2d: NumPy.Array
    :param model: cellposemodel for segmentation
    :type model: cellpose model
    :param channels: channels used for segmentation[description], defaults to [0, 0]
    :type channels: list, optional
    :param rescale: if diameter is set to None, and rescale is not None,
    then rescale is used instead of diameter for resizing image, defaults to None
    :type rescale: float, optional
    :param diameter: Estimated diameter of objects. If set to None,
    then diameter is automatically estimated if size model is loaded, defaults to None
    :type diameter: float, optional
    :param verbose: show additional output, defaults to False
    :type verbose: bool, optional
    :return: mask - binary mask
    :rtype: NumPy.Array
    """

    if not autotune:
        # Running performance tests to find the best convolution algorithm, this can take a while...
        # set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # define CHANNELS to run segmentation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0

    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # start the clock
    if verbose:
        start = perf_counter()

    # get the mask for a single image
    masks, _, _, _ = model.eval([image2d],
                                channels=channels,
                                diameter=diameter,
                                invert=False,
                                rescale=rescale,
                                do_3D=False,
                                net_avg=True,
                                tile=False,
                                flow_threshold=0.4,
                                cellprob_threshold=0.0,
                                progress=None)

    if verbose:
        end = perf_counter()
        st = end - start
        print('Segmentation Time CellPose:', st)

    return masks[0]


def segment_objects_cellpose2d(image2d: np.ndarray,
                               cp_model: models.CellposeModel,
                               channels: List[int] = [0, 0],
                               rescale: bool = None,
                               diameter: int = 17,
                               min_size: int = 15,
                               tile: bool = False,
                               tile_overlap: float = 0.1,
                               cellprob_threshold: float = 0.0):

    # get the mask for a single image
    masks, _, _ = cp_model.eval([image2d],
                                batch_size=8,
                                channels=channels,
                                diameter=diameter,
                                min_size=min_size,
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

    return masks[0]


def segment_nuclei_stardist(image2d, sdmodel,
                            prob_thresh=0.5,
                            overlap_thresh=0.3,
                            norm=True,
                            norm_pmin=1.0,
                            norm_pmax=99.8,
                            norm_clip=False):
    """[summary]

    :param image2d: 2d image to be segmented
    :type image2d: NumPy.Array
    :param sdmodel: stardit 2d model
    :type sdmodel: StarDist Model
    :param prob_thresh: probability threshold, defaults to 0.5
    :type prob_thresh: float, optional
    :param overlap_thresh: overlap threshold, defaults to 0.3
    :type overlap_thresh: float, optional
    :param norm: switch on image normalization, defaults to True
    :type norm: bool, optional
    :param norm_pmin: minimum percentile for normalization, defaults to 1.0
    :type norm_pmin: float, optional
    :param norm_pmax: maximum percentile for normalization, defaults to 99.8
    :type norm_pmax: float, optional
    :param norm_clip: clipping normalization, defaults to False
    :type norm_clip: bool, optional
    :return: mask - binary mask
    :rtype: NumPy.Array
    """

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # normalize image
    image2d_norm = normalize(image2d,
                             pmin=norm_pmin,
                             pmax=norm_pmax,
                             axis=None,
                             clip=norm_clip,
                             eps=1e-20,
                             dtype=np.float32)

    # predict the instances of th single nuclei
    mask2d, details = sdmodel.predict_instances(image2d_norm,
                                                axes=None,
                                                normalizer=None,
                                                prob_thresh=0.4,
                                                nms_thresh=0.3,
                                                n_tiles=None,
                                                show_tile_progress=True,
                                                overlap_label=None,
                                                verbose=False)

    return mask2d


# def set_device():
#     """Check if GPU working, and if so use it
#
#     :return: device - CPU or GPU
#     :rtype: mxnet device
#     """
#     # check if GPU working, and if so use it
#     use_gpu = utils.use_gpu()
#     print('Use GPU: ', use_gpu)
#
#     if use_gpu:
#         device = mxnet.gpu()
#     else:
#         device = mxnet.cpu()
#
#     return device


def load_cellpose_model(model_type='nuclei',
                        gpu=True,
                        net_avg=True):

    # load cellpose model for cell nuclei using GPU or CPU
    print('Loading Cellpose Model ...')

    model = models.Cellpose(gpu=gpu,
                            model_type=model_type,
                            net_avg=net_avg,
                            # torch=True
                            )

    return model


def load_cellpose_modelpath(model_path: List[str],
                            gpu: bool = True) -> models.CellposeModel:

    # load cellpose models
    print('Loading Cellpose Models from folder')

    model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)

    return model


def load_stardistmodel(modeltype='Versatile (fluorescent nuclei)'):

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # define and load the stardist model
    sdmodel = StarDist2D.from_pretrained(modeltype)

    return sdmodel


def stardistmodel_from_folder(modelfolder, mdname='2D_dsb2018'):

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sdmodel = StarDist2D(None, name=mdname, basedir=modelfolder)

    return sdmodel


def load_tfmodel(modelfolder='model_folder'):

    start = perf_counter()
    tfmodel = tf.keras.models.load_model(modelfolder)

    # Determine input shape required by the model and crop input image
    tile_height, tile_width = tfmodel.signatures["serving_default"].inputs[0].shape[1:3]
    end = perf_counter()
    print('Time to load TF2 model:', end - start)

    return tfmodel, tile_height, tile_width


def segment_zentf(image2d, model, classlabel=1):
    """Segment a singe [X, Y] 2D image using a pretrained segmentation
    model from the ZEN. The out will be a binary mask from the prediction
    of ZEN czmodel which is a TF.SavedModel with metainformation.
    Should work for any TF.SavedModel that fullfil the requirements.
    See also: https://pypi.org/project/czmodel/

    :param image2d: image to be segmented
    :type image2d: NumPy.Array
    :param model: trained TF2 model used for segmentation
    :type model: TF.SavedModel
    :param classlabel: Index for the class one is interested in
    :type classlabel: int
    :return: binary - binary mask of the specified class
    :rtype: NumPy.Array
    """

    # add add batch dimension (at the front) and channel dimension (at the end)
    image2d = image2d[np.newaxis, ..., np.newaxis]

    # Run prediction - array shape must be [1, 1024, 1024, 1]
    prediction = model.predict(image2d)[0]  # Removes batch dimension

    # Generate labels from one-hot encoded vectors
    prediction_labels = np.argmax(prediction, axis=-1)

    # get the desired class
    # background = 0, nuclei = 1 and borders = 2

    # extract desired class
    binary = np.where(prediction_labels == classlabel, 1, 0)

    return binary


def segment_zentf_tiling(image2d, model,
                         tilesize=1024,
                         classlabel=1,
                         overlap_factor=1):
    """Segment a singe [X, Y] 2D image using a pretrained segmentation
    model from the ZEN. The out will be a binary mask from the prediction
    of ZEN czmodel which is a TF.SavedModel with metainformation.

    Before the segmentation via the network will be applied
    the image2d will be tiled in order to match the tile size to the required
    batch tile size of the used network. Default is (1024, 1024)

    :param image2d: image to be segmented
    :type image2d: NumPy.Array
    :param model: trained TF2 model used for segmentation
    :type model: TF.SavedModel
    :param tilesize: required tile size for the segmentation model, defaults to 1024
    :type tilesize: int, optional
    :param classlabel: Index for the class one is interested in, defaults to 1
    :type classlabel: int, optional
    :param overlap_factor: overlap_factor of 2 = stride between each tile
    is only tile_shape/overlap_factor and therefore
    overlap_factor = 1 means no overlap, defaults to 1
    :type overlap_factor: int, optional
    :return: binary - binary mask of the specified class
    :rtype: Numpy.Array
    """

    # create tile image using MightMosaic
    image2d_tiled = MightyMosaic.from_array(image2d, (tilesize, tilesize),
                                            overlap_factor=overlap_factor,
                                            fill_mode='reflect')

    print('image2d_tiled shape : ', image2d_tiled.shape)
    # get number of tiles
    num_tiles = image2d_tiled.shape[0] * image2d_tiled.shape[1]
    print('Number of Tiles: ', num_tiles)

    # create array for the binary results
    binary_tiled = image2d_tiled

    ct = 0
    for n1 in range(image2d_tiled.shape[0]):
        for n2 in range(image2d_tiled.shape[1]):

            ct += 1
            print('Processing Tile : ', ct, ' Size : ', image2d_tiled[n1, n2, :, :].shape)

            # extract a tile
            tile = image2d_tiled[n1, n2, :, :]

            # get the binary from the prediction for a single tile
            binary_tile = segment_zentf(tile, model, classlabel=classlabel)

            # cats the result into the output array
            binary_tiled[n1, n2, :, :] = binary_tile

    # created fused binary and covert to int
    binary = binary_tiled.get_fusion().astype(int)

    return binary


def add_padding(image2d, input_height=1024, input_width=1024):
    """Add padding to an image if the size of that image is
    smaller than the required input width and input height

    :param image2d: 2d image
    :type image2d: NumPy.Array
    :param input_height: required height of the input image, defaults to 1024
    :type input_height: int, optional
    :param input_width: required width of the input image, defaults to 1024
    :type input_width: int, optional
    :return: image2d_padded - added image with teh required size
    :rtype: NumPy Array
    """

    if len(image2d.shape) == 2:
        isrgb = False
        image2d = image2d[..., np.newaxis]
    else:
        isrgb = True

    padding_height = input_height - image2d.shape[0]
    padding_width = input_width - image2d.shape[1]
    padding_left, padding_right = padding_width // 2, padding_width - padding_width // 2
    padding_top, padding_bottom = padding_height // 2, padding_height - padding_height // 2

    image2d_padded = np.pad(image2d, ((padding_top, padding_bottom),
                            (padding_left, padding_right), (0, 0)), 'reflect')

    if not isrgb:
        image2d_padded = np.squeeze(image2d_padded, axis=2)

    return image2d_padded, (padding_top, padding_bottom, padding_left, padding_right)


