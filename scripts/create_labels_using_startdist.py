# -*- coding: utf-8 -*-

#################################################################
# File        : create_labels_using_stardist.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from stardist.models import StarDist2D
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, segmentation
from skimage.transform import resize
from aicsimageio.writers import ome_tiff_writer as otw
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
import segmentation_tools as sgt
import segmentation_stardist as sg_sd


def process_labels(labels: np.ndarray,
                   seg_labeltype: str = "semantic",
                   do_area_filter: bool = True,
                   minsize: int = 200,
                   maxsize: int = 1000000,
                   do_erode: bool = False,
                   erode_numit: int = 1,
                   do_clear_borders: bool = False,
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    # find boundaries
    bounds = segmentation.find_boundaries(labels, connectivity=2, mode="thicker", background=0)

    # set all boundary pixel inside label image = Zero by inverting the boundary image
    new_labels = labels * ~bounds

    if verbose:
        show_plot(labels, ~bounds, new_labels, title1="labels", title2="~bounds", title3="new_labels")

    if do_area_filter:
        # filter labels by size
        new_labels, num_labels = sgt.area_filter(new_labels,
                                                 area_min=minsize,
                                                 area_max=maxsize)
        print("Area Filter - Objects:", num_labels)

    if do_erode:
        print("Erode labels Iteration:", erode_numit)
        new_labels = sgt.erode_labels(labels, erode_numit, relabel=True)

    if do_clear_borders:
        # clear border objects
        new_labels = segmentation.clear_border(new_labels)
        new_labels, num_labels = measure.label(new_labels > 0, return_num=True)
        print("Clear Borders - Objects:", num_labels)

        if verbose:
            show_plot(labels, ~bounds, new_labels, title1="labels", title2="~bounds", title3="new_labels")

    if seg_labeltype == "semantic":

        # set label value = 1
        sem_labels = np.where(new_labels > 0, 1, new_labels)

        # create the background by inverting the semantic labels
        background = 1 - sem_labels

        # make sure the sematic labels have the value = 255
        sem_labels = (sem_labels * 255).astype(np.uint8)
        background = (background * 255).astype(np.uint8)

        if verbose:
            show_plot(labels, sem_labels, background, title1="labels", title2="sem_labels", title3="background")

        return sem_labels, background

    if seg_labeltype == "instance":

        new_labels = new_labels.astype(np.uint16)
        background = 1 - (np.where(new_labels > 0, 1, new_labels))
        background = background.astype(np.uint8) * 255

        if verbose:
            show_plot(labels, new_labels, background, title1="labels", title2="inst_labels", title3="background")

        return new_labels, background


def show_plot(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray,
              title1: str = "img1",
              title2: str = "img2",
              title3: str = "img3",) -> None:

    # show the results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

    # show img1
    ax[0].imshow(img1, cmap="gray")
    ax[0].set_title(title1)

    # show img2
    ax[1].imshow(img2, cmap="gray")
    ax[1].set_title(title2)

    # show img3
    ax[2].imshow(img3, cmap="gray")
    ax[2].set_title(title3)

    plt.show()


def save_OMETIFF(img_FL: np.ndarray,
                 img_TL: np.ndarray,
                 new_labels: np.ndarray,
                 background: np.ndarray,
                 savepath_FL: str = "DAPI.ome.tiff",
                 savepath_TL: str = "PGC.ome.tiff",
                 savepath_NUC: str = "PGC_nuc.ome.tiff",
                 savepath_BGRD: str = "PGC_background.ome.tiff",
                 pixels_physical_sizes: List[float] = [1.0, 1.0, 1.0],
                 channel_names: Dict[str, str] = {"FL": "FL", "TL": "TL", "NUC": "NUC", "BGRD": "BGRD"}) -> None:

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_FL, savepath_FL,
                           channel_names=channel_names["FL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_TL, savepath_TL,
                           channel_names=channel_names["TL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the label
    otw.OmeTiffWriter.save(new_labels, savepath_NUC,
                           channel_names=channel_names["NUC"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the background
    otw.OmeTiffWriter.save(background, savepath_BGRD,
                           channel_names=["BGRD"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")


##########################################################################

basefolder = r"data"
#basefolder = r"D:\ImageData\Labeled_Datasets\DAPI_PGC\DAPI_PGC_20XNA095_stitched"
#basefolder = r"d:\ImageData\Labeled_Datasets\DAPI_PGC\DAPI_PGC_CD7_20XNA0.7\single"
dir_FL = os.path.join(basefolder, "fluo")
dir_LABEL = os.path.join(basefolder, "label")
dir_TL = os.path.join(basefolder, "trans")

os.makedirs(dir_FL, exist_ok=True)
os.makedirs(dir_LABEL, exist_ok=True)
os.makedirs(dir_TL, exist_ok=True)

suffix_orig = ".ome.tiff"
suffix_NUC = "_nuc.ome.tiff"
suffix_BGRD = "_background.ome.tiff"

# processing parameters
use_tiles = False
target_scaleXY = 0.5
rescale_image = False
tilesize_processing = 2000
min_borderwith_processing = 200

# get the desired StarDist2D model fur instance segmentation of cell nuclei
model = StarDist2D(None, name="2D_versatile_fluo", basedir="stardist_models")

# StarDist2D parameters
stardist_prob_thresh = 0.5
stardist_overlap_thresh = 0.3
stardist_overlap_label = None  # 0 is not supported yet
stardist_norm = True
stardist_norm_pmin = 1
stardist_norm_pmax = 99.8
stardist_norm_clip = False
n_tiles = None  # (4, 4)

# erode labels
do_erode = False
erode_numit = 3

# process the labels afterwards
do_area_filter = True
minsize_nuc = 20
maxsize_nuc = 5000
do_clear_borders = False

# define desired label output type
seg_labeltype = "semantic"
#seg_labeltype = "instance"

# CZI parameters
ext = ".czi"
ch_id_FL = 0  # channel index for the stained cell nuclei
ch_id_TL = 1  # channel index for the PGC or TL or ...

# for testing - show some plots
verbose = True

# iterating over all files
for file in os.listdir(basefolder):
    if file.endswith(ext):

        print("Processing CZI file:", file)

        cziname = file
        cziname_NUC = file[:-4] + "_onlyFL"
        cziname_TL = file[:-4] + "_onlyTL"

        # get the scaling from the CZI
        cziscale = czimd.CziScaling(os.path.join(basefolder, cziname))
        pixels_physical_sizes = [1, cziscale.X, cziscale.Y]

        scale_forward = target_scaleXY / cziscale.X
        new_shapeXY = int(np.round(tilesize_processing * scale_forward, 0))

        # open a CZI instance to read
        with pyczi.open_czi(os.path.join(basefolder, file)) as czidoc_r:

            if use_tiles:

                tilecounter = 0

                # create a "tile" by specifying the desired tile dimension and minimum required overlap
                tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=tilesize_processing,
                                                                  total_tile_height=tilesize_processing,
                                                                  min_border_width=min_borderwith_processing)

                # get the size of the bounding rectangle for the scene
                tiles = tiler.tile_rectangle(czidoc_r.scenes_bounding_rectangle[0])

                # show the created tile locations
                for tile in tiles:
                    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

                # loop over all tiles created by the "tiler"
                for tile in tqdm(tiles):

                    # read a specific tile from the CZI using the roi parameter
                    tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))[..., 0]
                    tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))[..., 0]

                    if rescale_image:
                        # scale the FL image to 0.5 micron per pixel (more or less)
                        tile2d_FL_scaled = resize(tile2d_FL, (new_shapeXY, new_shapeXY), preserve_range=True, anti_aliasing=True)

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels_scaled = sg_sd.segment_nuclei_stardist(tile2d_FL_scaled, model,
                                                                      prob_thresh=stardist_prob_thresh,
                                                                      overlap_thresh=stardist_overlap_thresh,
                                                                      overlap_label=stardist_overlap_label,
                                                                      n_tiles=n_tiles,
                                                                      norm_pmin=stardist_norm_pmin,
                                                                      norm_pmax=stardist_norm_pmax,
                                                                      norm_clip=stardist_norm_clip)

                        # scale the label image back to the original size preserving the label values
                        labels = resize(labels_scaled, (tilesize_processing, tilesize_processing), anti_aliasing=False, preserve_range=True).astype(np.uint32)

                    if not rescale_image:

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels = sg_sd.segment_nuclei_stardist(tile2d_FL, model,
                                                               prob_thresh=stardist_prob_thresh,
                                                               overlap_thresh=stardist_overlap_thresh,
                                                               overlap_label=stardist_overlap_label,
                                                               n_tiles=n_tiles,
                                                               norm_pmin=stardist_norm_pmin,
                                                               norm_pmax=stardist_norm_pmax,
                                                               norm_clip=stardist_norm_clip)

                    # process the labels
                    labels, background = process_labels(labels,
                                                        seg_labeltype=seg_labeltype,
                                                        do_area_filter=do_area_filter,
                                                        minsize=minsize_nuc,
                                                        maxsize=maxsize_nuc,
                                                        do_erode=do_erode,
                                                        erode_numit=erode_numit,
                                                        do_clear_borders=do_clear_borders,
                                                        verbose=verbose)

                    # save the original FL channel as OME-TIFF
                    savepath_FL = os.path.join(dir_FL, cziname_NUC + "_t" + str(tilecounter) + suffix_orig)

                    # save the original TL (PGC etc. ) channel as OME_TIFF
                    savepath_TL = os.path.join(dir_TL, cziname_TL + "_t" + str(tilecounter) + suffix_orig)

                    # save the labels for the nucleus and the background as OME-TIFF
                    savepath_BGRD = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_BGRD)
                    savepath_NUC = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_NUC)

                    # save the OME-TIFFs
                    save_OMETIFF(tile2d_FL, tile2d_TL, labels, background,
                                 savepath_FL=savepath_FL,
                                 savepath_TL=savepath_TL,
                                 savepath_NUC=savepath_NUC,
                                 savepath_BGRD=savepath_BGRD,
                                 pixels_physical_sizes=pixels_physical_sizes)

                    print("Saved images & labels for:", cziname_NUC, "tile:", tilecounter)

                    tilecounter += 1

            if not use_tiles:

                # read a specific tile from the CZI using the roi parameter
                tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL})[..., 0]
                tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL})[..., 0]

                if rescale_image:
                    # scale the FL image to 0.5 micron per pixel (more or less)
                    tile2d_FL_scaled = resize(tile2d_FL, (new_shapeXY, new_shapeXY), preserve_range=True, anti_aliasing=True)

                    # get the prediction for the current tile
                    # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                    labels_scaled = sg_sd.segment_nuclei_stardist(tile2d_FL_scaled, model,
                                                                  prob_thresh=stardist_prob_thresh,
                                                                  overlap_thresh=stardist_overlap_thresh,
                                                                  overlap_label=stardist_overlap_label,
                                                                  n_tiles=n_tiles,
                                                                  norm_pmin=stardist_norm_pmin,
                                                                  norm_pmax=stardist_norm_pmax,
                                                                  norm_clip=stardist_norm_clip)

                    # scale the label image back to the original size preserving the label values
                    labels = resize(labels_scaled, (tilesize_processing, tilesize_processing), anti_aliasing=False, preserve_range=True).astype(np.uint32)

                if not rescale_image:

                    # get the prediction for the current tile
                    # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                    labels = sg_sd.segment_nuclei_stardist(tile2d_FL, model,
                                                           prob_thresh=stardist_prob_thresh,
                                                           overlap_thresh=stardist_overlap_thresh,
                                                           overlap_label=stardist_overlap_label,
                                                           n_tiles=n_tiles,
                                                           norm_pmin=stardist_norm_pmin,
                                                           norm_pmax=stardist_norm_pmax,
                                                           norm_clip=stardist_norm_clip)

                    print("StarDist - OBjects:", labels.max())

                # process the labels
                labels, background = process_labels(labels,
                                                    seg_labeltype=seg_labeltype,
                                                    do_area_filter=do_area_filter,
                                                    minsize=minsize_nuc,
                                                    maxsize=maxsize_nuc,
                                                    do_erode=do_erode,
                                                    erode_numit=erode_numit,
                                                    do_clear_borders=do_clear_borders,
                                                    verbose=verbose)

                print("Saving images & labels for:", cziname_NUC)

                # save the original FL channel as OME-TIFF
                savepath_FL = os.path.join(dir_FL, cziname_NUC + suffix_orig)

                # save the original TL (PGC etc. ) channel as OME_TIFF
                savepath_TL = os.path.join(dir_TL, cziname_TL + suffix_orig)

                # save the labels for the nucleus and the background as OME-TIFF
                savepath_BGRD = os.path.join(dir_LABEL, cziname_TL + suffix_BGRD)
                savepath_NUC = os.path.join(dir_LABEL, cziname_TL + suffix_NUC)

                # save the OME-TIFFs
                save_OMETIFF(tile2d_FL, tile2d_TL, labels, background,
                             savepath_FL=savepath_FL,
                             savepath_TL=savepath_TL,
                             savepath_NUC=savepath_NUC,
                             savepath_BGRD=savepath_BGRD,
                             pixels_physical_sizes=pixels_physical_sizes)

                print("Saving finished for:", cziname_NUC)

    else:
        continue

print("Done.")
