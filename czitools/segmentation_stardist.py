# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_stardist.py
# Author      : sebi06
#
# Kudos also to: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/docs/demo.ipynb
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
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
from csbdeep.data import Normalizer, normalize_mi_ma


def segment_nuclei_stardist(img2d: np.ndarray,
                            sdmodel: StarDist2D,
                            axes: str = "YX",
                            prob_thresh: float = 0.5,
                            overlap_thresh: float = 0.3,
                            overlap_label: Union[int, None] = None,
                            blocksize: int = 1024,
                            min_overlap: int = 128,
                            n_tiles: Union[int, None] = None,
                            norm_pmin: float = 1.0,
                            norm_pmax: float = 99.8,
                            norm_clip: bool = False,
                            local_normalize: bool = True):
    """Segment cell nuclei using StarDist

    :param img2d: 2d image to be segmented
    :type img2d: NumPy.Array
    :param sdmodel: stardit 2d model
    :type sdmodel: StarDist Model
    :param axes: identifies for the image axes
    :type axes: str, optional
    :param prob_thresh: probability threshold, defaults to 0.5
    :type prob_thresh: float, optional
    :param overlap_thresh: overlap threshold, defaults to 0.3
    :type overlap_thresh: float, optional
    :param blocksize: Process input image in blocks of the provided shape, defaults to 1024
    :type blocksize: int, optional
    :param min_overlap: Amount of guaranteed overlap between blocks, defaults to 128
    :type min_overlap: int, optional
    :param n_tiles: number of tiles, e.g. (2, 2, 1), defaults to None
    :type n_tiles: tuple, optional
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

    if local_normalize:
        # normalize whole 2d image
        img2d = normalize(img2d,
                          pmin=norm_pmin,
                          pmax=norm_pmax,
                          axis=None,
                          clip=norm_clip,
                          eps=1e-20,
                          dtype=np.float32)

        normalizer = None

    if not local_normalize:
        mi, ma = np.percentile(img2d, [norm_pmin, norm_pmax])
        #mi, ma = image2d.min(), image2d.max()

        normalizer = MyNormalizer(mi, ma)

    # estimate blocksize
    max_dim_size = max(img2d.shape)
    blocksize = int(2 ** (np.round(np.log(max_dim_size)/np.log(2), 0) - 1))

    # define tiles
    if n_tiles is not None:
        if axes == "YX":
            tiles = (n_tiles, n_tiles)
        if axes == "YXC":
            tiles = (n_tiles, n_tiles, 1)
    if n_tiles is None:
        tiles = None

    # predict the instances of th single nuclei
    if max_dim_size >= 4096:
        mask2d, details = sdmodel.predict_instances_big(img2d,
                                                        axes=axes,
                                                        normalizer=normalizer,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=overlap_thresh,
                                                        block_size=blocksize,
                                                        min_overlap=min_overlap,
                                                        context=None,
                                                        n_tiles=tiles,
                                                        show_tile_progress=False,
                                                        overlap_label=overlap_label,
                                                        verbose=False)
    if max_dim_size < 4096:
        mask2d, details = sdmodel.predict_instances(img2d,
                                                    axes=axes,
                                                    normalizer=normalizer,
                                                    prob_thresh=prob_thresh,
                                                    nms_thresh=overlap_thresh,
                                                    n_tiles=tiles,
                                                    # n_tiles=None,
                                                    show_tile_progress=False,
                                                    overlap_label=overlap_label,
                                                    verbose=False)

    return mask2d


def load_stardistmodel(modeltype: str = 'Versatile (fluorescent nuclei)') -> StarDist2D:
    """Load an StarDist model from the web.

    Args:
        modeltype (str, optional): Name of the StarDist model to be loaded. Defaults to 'Versatile (fluorescent nuclei)'.

    Returns:
        StarDist2D: StarDist2D Model
    """

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # define and load the stardist model
    sdmodel = StarDist2D.from_pretrained(modeltype)

    return sdmodel


def stardistmodel_from_folder(modelfolder: str, mdname: str = '2D_dsb2018') -> StarDist2D:
    """Load an StarDist model from a folder.

    Args:
        modelfolder (str): Basefolder for the model folders.
        mdname (str, optional): Name of the StarDist model to be loaded. Defaults to '2D_dsb2018'.

    Returns:
        StarDist2D: StarDist2D Model
    """

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sdmodel = StarDist2D(None, name=mdname, basedir=modelfolder)

    return sdmodel


class MyNormalizer(Normalizer):

    def __init__(self, mi, ma):
        self.mi, self.ma = mi, ma

    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

    def after(*args, **kwargs):
        assert False

    @ property
    def do_after(self):
        return False
