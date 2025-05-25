"""
Source detection utilities for astronomical images.

This module provides functions for cleaning close coordinates and detecting
stars in images using thresholding and morphological operations.
"""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening


def clean_close_coords(coords, min_separation=30):
    """
    Remove coordinates that are closer than a minimum separation.

    Parameters
    ----------
    coords : np.ndarray
        Array of (x, y) coordinates.
    min_separation : float, optional
        Minimum allowed separation between coordinates.

    Returns
    -------
    np.ndarray
        Filtered array of coordinates with minimum separation enforced.
    """
    keep = {}
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    distances[np.eye(len(coords), dtype=bool)] = np.nan

    for i in range(len(coords)):
        keep[i] = True

    for i in range(len(coords)):
        if keep[i]:
            distance = distances[i]
            idxs = np.where(distance < min_separation)[0]
            for j in idxs[idxs > i]:
                keep[int(j)] = False

    return np.array([coords[i] for i in range(len(coords)) if keep[i]])


def stars_detection(image, threshold=5, opening=5):
    """
    Detect stars in an image using thresholding and morphological opening.

    Parameters
    ----------
    image : np.ndarray or object with .data attribute
        2D image data or object with .data attribute.
    threshold : float, optional
        Detection threshold in units of standard deviation above the median.
    opening : int, optional
        Size of the morphological opening kernel.

    Returns
    -------
    list
        List of skimage.measure._regionprops.RegionProperties objects for detected regions.
    """
    opening_kernel = np.ones((opening, opening))
    flat_image = image.flatten()
    median = np.nanmedian(flat_image)
    flat_image = flat_image[np.abs(flat_image - median) < np.nanstd(flat_image)]
    threshold = threshold * np.nanstd(flat_image) + median
    regions = regionprops(
        label(binary_opening(image.data > threshold, opening_kernel)), image
    )
    regions = sorted(regions, key=lambda x: x.intensity_max, reverse=True)
    return regions
