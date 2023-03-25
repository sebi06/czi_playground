# -*- coding: utf-8 -*-

#################################################################
# File        : run_cellpose.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import os
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from dataclasses import dataclass, field
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
import pandas as pd
from skimage import measure, segmentation
from skimage.measure import regionprops
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
from tqdm.contrib.itertools import product
from cellpose import models


def segment_nuclei_cellpose2d(image2d, cp_model,
                              channels=[0, 0],
                              rescale=None,
                              diameter=None,
                              min_size=15,
                              tile=False,
                              tile_overlap=0.1,
                              verbose=False,
                              cellprob_threshold=0.0):
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

    # get the mask for a single image
    masks, _, _, _ = cp_model.eval([image2d],
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


def area_filter(im: np.ndarray,
                area_min: int = 10,
                area_max: int = 100000) -> np.ndarray:

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
    im_filt = measure.label(im_approved > 0)

    return im_filt


def show_plot(img1: np.ndarray, img2: np.ndarray) -> None:

    # show the results
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].imshow(img1, cmap="gray")
    ax[1].imshow(img2, cmap="gray")
    plt.show()


#############################################################

# select plotting backend
plt.switch_backend('Qt5Agg')
verbose = False

# define the filename
image_path = r"data/A01.czi"
#image_path = r"data/CH=3.czi"
savepath = misc.get_fname_woext(image_path) + "_segCP.czi"

# get the metadata
mdata = czimd.CziMetadata(image_path)


chindex_seg = 0  # channel containing the objects, e.g. the nuclei
minsize_nuc = 100  # minimum object size [pixel]
maxsize_nuc = 5000  # maximum object size [pixel]
channels = [0, 0]  # when applying it to a single image
diameter = 17
flatten_labels = False
cellprob_threshold = 0.0
tile = False
tile_overlap = 0.1
verbose = True

# define model type
model_type = "nuclei"
#model_type = "cyto"

models_nuclei = [os.path.join(os.getcwd(), os.path.join(".cellpose", "nucleitorch_0")),
                 os.path.join(os.getcwd(), os.path.join(".cellpose", "nucleitorch_1")),
                 os.path.join(os.getcwd(), os.path.join(".cellpose", "nucleitorch_2")),
                 os.path.join(os.getcwd(), os.path.join(".cellpose", "nucleitorch_3"))]

models_cyto = [os.path.join(os.getcwd(), os.path.join(".cellpose", "cytotorch_0")),
               os.path.join(os.getcwd(), os.path.join(".cellpose", "cytotorch_1")),
               os.path.join(os.getcwd(), os.path.join(".cellpose", "cytotorch_2")),
               os.path.join(os.getcwd(), os.path.join(".cellpose", "cytotorch_3"))]

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)
results = pd.DataFrame()

# measure region properties
to_measure = ('label',
              'area',
              'centroid',
              'max_intensity',
              'mean_intensity',
              'min_intensity',
              'bbox')

# load the cellpose model
#cp_model = load_cellpose_model(model_type=model_type, gpu=True, net_avg=True)

cp_model = load_cellpose_modelpath(models_nuclei, gpu=True)


with pyczi.create_czi(savepath, exist_ok=True) as czidoc_w:

    with pyczi.open_czi(image_path) as czidoc_r:

        # check if dimensions are None (because the do not exist for that image)
        sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        # read array for the scene
        for s, t, z, in product(range(sizeS),
                                range(sizeT),
                                range(sizeZ)):

            values = {'S': s, 'T': t, 'Z': z, 'C': chindex_seg, 'Number': 0}

            # read 2D plane in case there are (no) scenes
            if mdata.image.SizeS is None:
                img2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': chindex_seg})[..., 0]
            else:
                img2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': chindex_seg}, scene=s)[..., 0]

            dtype_orig = img2d.dtype

            # get the mask for the current image
            labels = segment_objects_cellpose2d(img2d, cp_model,
                                                rescale=None,
                                                channels=channels,
                                                diameter=diameter,
                                                min_size=minsize_nuc,
                                                tile=tile,
                                                tile_overlap=tile_overlap,
                                                cellprob_threshold=cellprob_threshold)

            # # get the mask for the current image
            # labels = segment_nuclei_cellpose2d(img2d, cp_model,
            #                                     rescale=None,
            #                                     channels=channels,
            #                                     diameter=diameter,
            #                                     min_size=minsize_nuc,
            #                                     tile=tile,
            #                                     tile_overlap=tile_overlap,
            #                                     cellprob_threshold=cellprob_threshold,
            #                                     verbose=True)

            print("CellPose - OBjects:", labels.max())

            # find boundaries
            bound = segmentation.find_boundaries(
                labels, connectivity=2, mode="thicker", background=0)

            # set all boundary pixel inside label image = Zero by inverting the boundary image
            labels = labels * ~bound

            if verbose:
                show_plot(img2d, labels)

            # filter labels by size
            labels = area_filter(labels,
                                 area_min=minsize_nuc,
                                 area_max=maxsize_nuc)

            # clear border objects
            labels = segmentation.clear_border(labels)

            # measure the specified parameters store in a dataframe
            props = pd.DataFrame(measure.regionprops_table(labels,
                                                           intensity_image=img2d,
                                                           properties=to_measure)
                                 ).set_index('label')

            # filter objects by size
            #props = props[(props['area'] >= minsize_nuc) & (props['area'] <= maxsize_nuc)]

            # add well information for CZI metadata
            try:
                props['WellId'] = mdata.sample.well_array_names[s]
                props['Well_ColId'] = mdata.sample.well_colID[s]
                props['Well_RowId'] = mdata.sample.well_rowID[s]
            except (IndexError, KeyError) as error:
                print('Error:', error)
                print('Well Information not found. Using S-Index.')
                props['WellId'] = s
                props['Well_ColId'] = s
                props['Well_RowId'] = s

            # add plane indices
            props['S'] = s
            props['T'] = t
            props['Z'] = z
            props['C'] = chindex_seg

            # count the number of objects
            values['Number'] = props.shape[0]
            if verbose:
                print('Well:', props['WellId'].iloc[0], ' Objects: ', values['Number'])

            # update dataframe containing the number of objects
            objects = pd.concat([objects, pd.DataFrame(values, index=[0])], ignore_index=True)
            results = pd.concat([results, props], ignore_index=True)

            # add dimension for CZI pixel type at the end of array - [Y, X, 1]
            labels = labels[..., np.newaxis]

            # convert to desired dtype in place
            labels = labels.astype(dtype_orig, copy=False)

            if flatten_labels:
                labels[labels > 0] = 255

            # write the label image to CZI
            if mdata.image.SizeS is None:

                # write 2D plane in case of no scenes
                czidoc_w.write(labels, plane={"T": t,
                                              "Z": z,
                                              "C": chindex_seg})
            else:
                # write 2D plane in case scenes exist
                czidoc_w.write(labels, plane={"T": t,
                                              "Z": z,
                                              "C": chindex_seg},
                               scene=s,
                               location=(czidoc_r.scenes_bounding_rectangle[s].x,
                                         czidoc_r.scenes_bounding_rectangle[s].y)
                               )

    # reorder dataframe with single objects
    new_order = list(results.columns[-7:]) + list(results.columns[:-7])
    results = results.reindex(columns=new_order)

    # define name for CSV tables
    obj_csv = misc.get_fname_woext(image_path) + '_obj.csv'
    objparams_csv = misc.get_fname_woext(image_path) + '_objparams.csv'

    # save the DataFrames as CSV tables
    objects.to_csv(obj_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Table as CSV :', obj_csv)

    results.to_csv(objparams_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Parameters Table as CSV :', objparams_csv)
    print('Segmentation done.')
