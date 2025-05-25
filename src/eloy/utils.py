"""
General utilities for astronomical image analysis.

This module provides helper functions for cutouts, statistical metrics,
binning, and generating synthetic images.
"""

from astropy.nddata import Cutout2D
import numpy as np
from astropy.nddata import Cutout2D


def cutout(data, coords, shape, wcs=None):
    """
    Extract cutouts from data at given coordinates.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    coords : array-like
        List of (x, y) coordinates.
    shape : tuple
        Shape of the cutout.
    wcs : WCS or None, optional
        World Coordinate System object.

    Returns
    -------
    np.ndarray
        Array of cutout images.
    """
    values = []
    for coords in coords:
        cutout = Cutout2D(
            data, coords, shape, wcs=wcs, fill_value=np.nan, mode="partial"
        )
        values.append(cutout.data)
    return np.array(values)


def std_diff_metric(fluxes):
    """
    Compute the standard deviation of the difference along the last axis.

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes array.

    Returns
    -------
    np.ndarray
        Standard deviation of the differences.
    """
    k = len(list(np.shape(fluxes)))
    return np.std(np.diff(fluxes, axis=k - 1), axis=k - 1)


def stability_aperture(fluxes):
    """
    Compute the mean absolute difference between consecutive fluxes.

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes array.

    Returns
    -------
    np.ndarray
        Mean absolute difference for each aperture.
    """
    lc_c = np.abs(np.diff(fluxes, axis=0))
    return np.mean(lc_c, axis=1)


def index_binning(x, size):
    """
    Bin indices of x into bins of given size.

    Parameters
    ----------
    x : array-like
        Array to bin.
    size : int or float
        Bin size.

    Returns
    -------
    list
        List of arrays of indices for each bin.
    """
    if isinstance(size, float):
        bins = np.arange(np.min(x), np.max(x), size)
    else:
        x = np.arange(0, len(x))
        bins = np.arange(0.0, len(x), size)

    d = np.digitize(x, bins)
    n = np.max(d) + 2
    indexes = []

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            indexes.append(s)

    return indexes


def binned_nanstd(x, bins: int = 12):
    """
    Return a function to compute the mean of the standard deviation in bins.

    Parameters
    ----------
    x : np.ndarray
        Array to bin.
    bins : int, optional
        Number of bins.

    Returns
    -------
    callable
        Function that computes the mean of the standard deviation in bins.
    """
    # set binning idxs for white noise evaluation
    bins = np.min([x.shape[-1], bins])
    n = x.shape[-1] // bins
    idxs = np.arange(n * bins)

    def compute(f):
        return np.nanmean(
            np.nanstd(np.array(np.split(f.take(idxs, axis=-1), n, axis=-1)), axis=-1),
            axis=0,
        )

    return compute


def fake_image(shape=50, seed=0, stars=30):
    """
    Generate a fake image with random stars.

    Parameters
    ----------
    shape : int, optional
        Size of the image (shape x shape).
    seed : int, optional
        Random seed.
    stars : int, optional
        Number of stars.

    Returns
    -------
    tuple
        Tuple (image, coords) where image is the generated image and coords are the star positions.
    """
    np.random.seed(seed)
    image = np.zeros((shape, shape))
    coords = (np.random.rand(2, stars) * shape).astype(int).T
    for i, j in coords:
        image[j, i] = 1.0
    return image, coords
