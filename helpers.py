#!/usr/bin/env python

"""Contains base-level functions that are required for the others to run."""
import numpy as np
import os
import re


def frac_round(number, frac, centre_frac):
    """Converts fraction 'frac' to crop a dimension of an image by, say x,
    centred at centre_frac, to an integer index for the array with 'number'
    indices along that axis. Refer to crop_array for more info."""
    lower_bound = (number/2.*(1-frac)) + (number * float(centre_frac))
    upper_bound = (number/2.*(1+frac)) + (number * float(centre_frac))

    return int(np.round(lower_bound)), int(np.round(upper_bound))


def verify_vector(vector):
    """Checks the input vector has 3 components."""
    r = np.array(vector)
    assert r.shape == (3,), "The variable 'vector' must have 3 components."
    return r


def ccf(list_int, n):
    """Returns the common factor of the integers in the list list_int that
    is closest to n. If two are equally close, the smallest is returned."""
    all_factors = [set(_factors(num)) for num in list_int]
    common_factors = list(set(all_factors[0]).intersection(*all_factors[1:]))
    return _one_disallowed(common_factors, n)


def closest_factor(f, n):
    """Returns the factor of f that is closest to n. If n is equidistant
    from two factors of f, the smallest of the two factors is returned."""
    return _one_disallowed(_factors(f), n)


def unchanged(*args):
    """Returns the arguments; the default function for image
    post-processing to return the input array unchanged."""
    return args


def _one_disallowed(factors, n):
    """For a list of integers 'factors' in ascending order, return the
    number closest to n (choose the smallest if 2 are equidistant) as long as
    it is not 1. If it is 1, return the second closest factor."""
    closest = min(factors, key=lambda x: abs(x - n))
    try:
        return closest if closest != 1 else factors[1]
    except:
        raise Exception('Only common factor is 1. Crop or zero-pad the image '
                        'before down-sampling.')


def _factors(num):
    """Returns the factors of a number, in ascending order as a list."""
    factor_list = list(reduce(list.__add__, ([j, num // j] for j in range(
        1, int(num ** 0.5) + 1) if num % j == 0)))
    factor_list.sort()
    return factor_list


def bake(fun, args=None, kwargs=None, position_to_pass_through=0):
    """Returns an object given by the function 'fun' with its arguments,
    known as a curried function or closure. These objects can be passed into
    other functions to be evaluated.

    :param fun: The function object without any arguments specified.
    :param args: A list of the positional arguments.
    :param kwargs: A list of keyword arguments.
    :param position_to_pass_through: See docstring for 'wrapped'.
    :return: The object containing the function with its arguments."""

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    def wrapped(image):
        """Parameter position_to_pass_through specifies the index of the
        parameter 'image' in the sequence of positional arguments for 'fun'."""
        return fun(*(args[:position_to_pass_through] + [image] + args[(
            position_to_pass_through+1):]), **kwargs)

    return wrapped


def make_dirs(file_path):
    """Extract directory structure from file path and create the directories if
    necessary.
    :param file_path: String for the relative or absolute file path."""
    if file_path[0] == '/':
        # Allows for Linux ABSOLUTE path format.
        path = file_path.split('/')[:-1]
    # The following 2 are relative file paths.
    elif file_path[:2] == './':
        path = file_path.split('/')[1:-1]
    elif re.match(r'\w+', file_path, flags=re.IGNORECASE):
        path = file_path.split('/')[:-1]
    else:
        raise ValueError('The file path has incorrect format.')

    if len(path) >= 1 and '' not in path:
        dir_path = '/'.join(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return


def add_extension(string, extension='.hdf5'):
    """Checks if 'string' has 'extension' at end, and appends it if not."""
    if not re.search(r'{}'.format(extension), string.strip()):
        string += extension
    return string


def positions_maker(x=np.array([0]), y=np.array([0]), z=np.array([0]),
                    initial_pos=np.array([0, 0, 0])):
    """Generator to produce N x 3 array of all possible permutations of 1D
    arrays x and y, such that N = len(x) * len(y). For example x = [1,2] and
    y = [3,4] yields [1, 3, 0], [1, 4, 0], [2, 3, 0], [2, 4, 0] respectively.
    This is added to [0, 0, 0] before being output."""
    i = 0
    while i < x .size:
        j = 0
        while j < y.size:
            k = 0
            while k < z.size:
                yield np.array([x[i], y[j], z[k]]) + initial_pos
                k += 1
            j += 1
        i += 1

