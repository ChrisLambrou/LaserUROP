#!/usr/bin/env python

"""measurements.py
Contains functions to perform single measurements on BGR/greyscale arrays
that have been appropriately processed."""

import numpy as np
from scipy import ndimage as sn

import image_proc as proc


def sharpness_lap(bgr_array):
    """Calculate sharpness as the Laplacian of the BGR image array.
    :param bgr_array: The 3-channel image to calculate sharpness for.
    :return: The mean Laplacian.
    """
    image_bw = np.mean(bgr_array, 2)
    image_laplace = sn.filters.laplace(image_bw)
    return np.mean(np.abs(image_laplace))


def sharpness_clippedlog(bgr_array):
    """Calculate sharpness as the clipped LoG of the BGR image array.
    :param bgr_array: The 3-channel image to calculate sharpness for.
    :return: The mean clipped LoG value across all pixels in the image.
    """
    image_bw = np.mean(bgr_array, 2)
    image_filtered = sn.filters.gaussian_laplace(image_bw, 1)

    # We use the -ve values and invert the result, because we
    # care about pixels where the intensity value is lower than their
    # surroundings, dark bacteria with a lighter halo, and want to exclude
    # lighter bacteria with a darker halo.
    return -np.mean(np.clip(image_filtered, -10000000, 0))


def brightness(arr):
    """Calculates the mean brightness of an array.
    :param arr: A BGR or greyscale array.
    :return: The scalar brightness value."""

    # If the array is BGR, convert to greyscale before calculating brightness.
    if len(arr.shape) == 3:
        arr = proc.make_greyscale(arr, greyscale=True)
    elif len(arr.shape) != 2:
        raise ValueError('Array has invalid shape: {}'.format(arr.shape))

    return np.mean(arr)


