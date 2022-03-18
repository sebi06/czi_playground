# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_stardist.py
# Version     : 0.1
# Author      : sebi06
# Date        : 18.03.2021
#
# Kudos also to: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/docs/demo.ipynb
#
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import numpy as np
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize


def segment_nuclei_stardist(image2d, sdmodel,
                            prob_thresh=0.5,
                            overlap_thresh=0.3,
                            overlap_label=None,
                            norm_pmin=1.0,
                            norm_pmax=99.8,
                            norm_clip=False):
    """Segment cell nuclei using StarDist

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
                                                prob_thresh=prob_thresh,
                                                nms_thresh=overlap_thresh,
                                                n_tiles=None,
                                                show_tile_progress=True,
                                                overlap_label=overlap_label,
                                                verbose=False)

    return mask2d


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
