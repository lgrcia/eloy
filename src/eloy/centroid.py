"""
Centroiding utilities for astronomical images.

This module provides functions to compute centroids of sources in images
using photutils, with robust handling of edge cases.
"""

from photutils.centroids import (
    centroid_quadratic,
    centroid_sources,
)
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from eloy import utils

default_centroid_func = centroid_quadratic


def photutils_centroid(data, coords, cutout=21, centroid_fun=None):
    """
    Compute centroids for a list of coordinates using photutils.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    coords : np.ndarray
        Array of (x, y) coordinates.
    cutout : int, optional
        Size of the cutout box for centroiding.
    centroid_fun : callable or None
        Centroiding function to use.

    Returns
    -------
    np.ndarray
        Array of centroid coordinates.
    """
    if centroid_fun is None:
        centroid_fun = default_centroid_func

    in_image = np.all(coords < np.array(data.shape[::-1]) - (1, 1), axis=1)
    in_image = np.logical_and(in_image, np.all(coords > (0, 0), axis=1))

    x, y = coords[in_image].T.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        in_image_centroid_coords = np.array(
            centroid_sources(data, x, y, box_size=cutout, centroid_func=centroid_fun)
        ).T
    centroid_coords = coords.copy()
    centroid_coords[in_image] = in_image_centroid_coords
    idxs = np.flatnonzero(~np.all(np.isfinite(centroid_coords), 1))
    centroid_coords[idxs] = coords[idxs]
    return centroid_coords


def ballet_centroid(data, coords, cnn):
    cutouts = utils.cutout(data, coords, (15, 15))
    return coords - 15 / 2 + cnn.centroid(cutouts)
