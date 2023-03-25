# -*- coding: utf-8 -*-

#################################################################
# File        : processing_tools.py
# Version     : 0.0.2
# Author      : sebi06
# Date        : 02.11.2021
#
# Disclaimer: This tool is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def calc_normvar(img2d: np.ndarray) -> float:
    """Determine normalized focus value for a 2D image
    - based on algorithm F - 11 "Normalized Variance"
    - Taken from: Sun et al., 2004. MICROSCOPY RESEARCH AND TECHNIQUE 65, 139–149.
    - Maximum value is best-focused, decreasing as defocus increases

    :param img2d: 2D image
    :type img2d: NumPy.Array
    :return: normalized focus value for the 2D image
    :rtype: float
    """

    # get the mean intensity value for the plane
    mean = np.mean(img2d)

    # get height and width of the plane
    height = img2d.shape[0]
    width = img2d.shape[1]

    # subtract the mean and sum up the whole array
    fi = (img2d - mean)**2
    b = np.sum(fi)

    # calculate the normalized variance value
    normvar = b / (height * width * mean)

    return normvar


def get_sharpest_plane(zstack: np.ndarray) -> Tuple[np.ndarray, list]:
    """Get the sharpest plane from az-stack with shape [z, x, y]

    :param zstack: 3D stack
    :type zstack: NumPy.Array
    :return: tuple (sharpest_plane from zstack, list of focus values
    :rtype: tuple(NumPy.Array, fv)
    """

    # create empty list of focus values and set initial value to zero
    fv = []
    fv_max = 0

    # loop over all zplanes
    for z in range(zstack.shape[0]):

        # get the normalizes variance for the plane
        normvar_curr = calc_normvar(zstack[z, :, :])

        # append value to the list of focus values
        fv.append(normvar_curr)

        # update the maximum focus value and sharpest plane
        if normvar_curr > fv_max:
            fv_max = normvar_curr
            sharpest_plane = zstack[z, :, :]

    return (sharpest_plane, fv)


