#!/usr/bin/env python

"""image_proc.py
Functions to process and measure image features."""

import numpy as np
import gen_helpers as h
from image_mmts import get_res


def crop_section(bgr_arr, frac, centre_frac=(0, 0)):
    """Crops the central portion of the image by a specified amount.
    :param bgr_arr: The 3D image array to split, in the format (no. of row
    pixels in image, no. of column pixels in image, 3 BGR values).
    :param frac: A tuple with the percentage of the image along (x,
    y) to retain. For example, x_frac = 30, y_frac = 50 would result in a
    cropped image 15% either side of the centre along x and 25% either side of
    the centre along y.
    :param centre_frac: The centre of the cropped image relative to the main
    image, as a fraction of the (x, y) length of the main image with origin
    at (0, 0). For example, (1/2., 1/4.) would result in the cropped image
    being centred on the top edge of the main image, 3/4 of the way along
    the edge from the top left corner. Checks exist to ensure the crop
    covers only the range of the main image.
    :return: The cropped image BGR array."""

    (x_res, y_res) = get_res(bgr_arr)[:2]
    if type(frac) is not tuple:
        frac = (frac, frac)
    for each in frac:
        assert each >= 0, "{} is an invalid fraction of the image to " \
                          "crop.".format(each)
    for each in centre_frac:
        assert -1/2. <= each <= 1/2., "Centre lies outside range of image."

    crop = bgr_arr[_frac_round(y_res, frac[1], centre_frac[1])[0]:
                   _frac_round(y_res, frac[1], centre_frac[1])[1],
                   _frac_round(x_res, frac[0], centre_frac[0])[0]:
                   _frac_round(x_res, frac[0], centre_frac[0])[1], :]

    actual_fraction = float(crop.size)/bgr_arr.size * 100
    print r'Cropped the centre_frac {}% of image.'.format(actual_fraction)
    return crop, actual_fraction


def crop_img_into_n(bgr_arr, n):
    """Splits the bgr array into n equal sized chunks by array slicing.
    :param bgr_arr: The 3D array to split, in the format (no. of row pixels in
    image, no. of column pixels in image, 3 BGR values).
    :param n: The number of equal sized chunks to split the array into.
    :return: A list in the format [tuple of resolution of main image,
    tuple of number of subimages per (row, column), list of lists of each
    sub-image array."""

    [x_res, y_res, tot_res] = get_res(bgr_arr)

    # Round n to the nearest factor of the total resolution so the image is
    # cropped while maintaining the same aspect ratio per crop.
    num_subimages = h.closest_factor(tot_res, n)
    print "Splitting image into {} sub-images.".format(num_subimages)

    [x_subimgs, y_subimgs] = _get_num_subimages((x_res, y_res), num_subimages)
    pixel_step = _get_pixel_step((x_res, y_res), (x_subimgs, y_subimgs))

    # Split image along y, then x. Lists have been used here instead of
    # arrays because although it may be slower, memory leaks are less likely.
    split_y = np.split(bgr_arr, y_subimgs, axis=0)
    split_xy = []
    for row in split_y:
        split_x = np.split(row, x_subimgs, axis=1)
        split_xy.append(split_x)

    # split_xy is a list of lists containing subsections of the 3D array
    # bgr_arr.
    return [pixel_step, split_xy]


def down_sample(array, factor_int):
    """Down sample a numpy array, such that the total number of pixels is
    reduced by a factor of factor_int**2. This is done by grouping the
    pixels into square blocks and taking the average of each of the B, G, R
    values in each block.
    :return: The down sampled array."""

    # Ensure that factor_int divides into the number of pixels on each side
    # - if not, round to the nearest factor.
    factor_int = h.ccf(array.shape[0:2], factor_int)
    print "Using {} x {} pixel blocks for down-sampling.".format(factor_int,
                                                                 factor_int)
    bin_y = np.mean(np.reshape(
        array, (array.shape[0], array.shape[1]/factor_int, factor_int, 3),
        order='C'), axis=2, dtype=np.uint16)
    binned = np.mean(np.reshape(bin_y, (factor_int, array.shape[0]/factor_int,
        array.shape[1]/factor_int, 3), order='F'), axis=0, dtype=np.uint16)
    return binned


def _frac_round(number, frac, centre_frac):
    """Function to aid readability, used in crop_section. Note that frac and
    centre_frac are individual elements of the tuples defined in
    crop_section."""
    frac /= 100.
    lower_bound = (number/2.*(1-frac)) + (number * float(centre_frac))
    upper_bound = (number/2.*(1+frac)) + (number * float(centre_frac))

    if lower_bound < 0:
        lower_bound = 0
        raise Warning('Lower bound of cropped image exceeds main image '
                      'dimensions. Setting it to start of main image.')
    if upper_bound > number:
        upper_bound = 1
        raise Warning('Upper bound of cropped image exceeds main image '
                      'dimensions. Setting it to end of main image.')
    return int(np.round(lower_bound)), int(np.round(upper_bound))


def _get_pixel_step(res, num_sub_imgs):
    """Calculates the number of pixels per subimage along x and y.
    :param res: A tuple of the resolution of the main image along (x, y).
    :param num_sub_imgs: A tuple of no. of subimages along x, y e.g. (4, 3).
    :return: A tuple of (number of x pixels per sub-image, number of y
    pixels per sub-image)."""
    return res[0] / num_sub_imgs[0], res[1] / num_sub_imgs[1]


def _get_num_subimages(res, tot_subimages):
    """Returns a tuple of the number of subimages along x and y such that
    aspect ratio is maintained.
    :param res: (x_resolution, y_resolution).
    :param tot_subimages: Total number of subimages to split the main image
    into."""
    x_split = np.sqrt(res[0] / res[1] * tot_subimages)
    y_split = tot_subimages / x_split
    return x_split, y_split
