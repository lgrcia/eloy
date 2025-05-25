"""
Calibration routines for astronomical images.

This module provides functions to compute master bias, dark, and flat frames,
as well as to calibrate raw image data using these frames.
"""

import numpy as np
from astropy.io import fits


def divisors(n):
    """
    Find all divisors of an integer.

    Parameters
    ----------
    n : int
        The integer to find divisors for.

    Returns
    -------
    np.ndarray
        Array of divisors of n.
    """
    _divisors = []
    i = 1
    while i <= n:
        if n % i == 0:
            _divisors.append(i)
        i = i + 1
    return np.array(_divisors)


def easy_median(images, m=50):
    """
    Compute the median of images in chunks to avoid memory errors.

    Parameters
    ----------
    images : array-like
        List or array of images.
    m : int, optional
        Chunk size for splitting the computation.

    Returns
    -------
    np.ndarray
        Concatenated median of the images.
    """
    # To avoid memory errors, we split the median computation in 50
    images = np.array(images)
    shape_divisors = divisors(images.shape[1])
    n = shape_divisors[np.argmin(np.abs(m - shape_divisors))]
    return np.concatenate(
        [np.nanmedian(im, axis=0) for im in np.split(images, n, axis=1)]
    )


default_fun_load = lambda file: fits.open(file)[0].data
default_fun_exp = lambda file: fits.open(file)[0].header["EXPTIME"]


def master_dark(bias=None, files=None, fun_load=None, fun_exp=None):
    """
    Create a master dark frame from a list of dark files.

    Parameters
    ----------
    bias : np.ndarray or None
        Master bias frame to subtract.
    files : list or None
        List of file paths to dark frames.
    fun_load : callable or None
        Function to load image data from file.
    fun_exp : callable or None
        Function to extract exposure time from file.

    Returns
    -------
    np.ndarray
        Master dark frame.
    """
    if bias is None:
        bias = master_bias()
    if fun_load is None:
        fun_load = default_fun_load
    if fun_exp is None:
        fun_exp = default_fun_exp

    if files is None:
        return np.array([0.0])
    else:
        _darks = []
        for file in files:
            data = fun_load(file)
            exposure = fun_exp(file)
            _darks.append((data - bias) / exposure)
            del data
        master = easy_median(_darks)
        del _darks
        return master


def master_flat(bias=None, dark=None, files=None, fun_load=None, fun_exp=None):
    """
    Create a master flat frame from a list of flat files.

    Parameters
    ----------
    bias : np.ndarray or None
        Master bias frame to subtract.
    dark : np.ndarray or None
        Master dark frame to subtract.
    files : list or None
        List of file paths to flat frames.
    fun_load : callable or None
        Function to load image data from file.
    fun_exp : callable or None
        Function to extract exposure time from file.

    Returns
    -------
    np.ndarray
        Master flat frame.
    """
    if fun_load is None:
        fun_load = default_fun_load
    if fun_exp is None:
        fun_exp = default_fun_exp
    if bias is None:
        bias = master_bias()
    if dark is None:
        dark = master_dark()

    if files is None:
        return np.array([1.0])
    else:
        _flats = []
        for file in files:
            data = fun_load(file)
            exposure = fun_exp(file)
            _flat = data - bias - dark * exposure
            _flat /= np.mean(_flat)
            _flats.append(_flat)
            del data
        master = easy_median(_flats)
        del _flats
        return master


def master_bias(files=None, fun_load=None):
    """
    Create a master bias frame from a list of bias files.

    Parameters
    ----------
    files : list or None
        List of file paths to bias frames.
    fun_load : callable or None
        Function to load image data from file.

    Returns
    -------
    np.ndarray
        Master bias frame.
    """
    if fun_load is None:
        fun_load = default_fun_load

    if files is None:
        return np.array([0.0])
    else:
        _biases = []
        for file in files:
            data = fun_load(file)
            _biases.append(data)
            del data
        master = easy_median(_biases)
        del _biases
        return master


def calibrate(data, exposure, dark, flat, bias):
    """
    Calibrate raw image data using master calibration frames.

    Parameters
    ----------
    data : np.ndarray
        Raw image data.
    exposure : float
        Exposure time of the image.
    dark : np.ndarray
        Master dark frame.
    flat : np.ndarray
        Master flat frame.
    bias : np.ndarray
        Master bias frame.

    Returns
    -------
    np.ndarray
        Calibrated image data.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        calibrated_data = (data - (dark * exposure + bias)) / flat

    calibrated_data[calibrated_data < 0] = np.nan
    calibrated_data[~np.isfinite(calibrated_data)] = -1
    return calibrated_data
