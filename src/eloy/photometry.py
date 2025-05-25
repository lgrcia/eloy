"""
Aperture photometry utilities for astronomical images.

This module provides functions for performing aperture photometry and
estimating background using annular sigma-clipping.
"""

from photutils.aperture import aperture_photometry as photutils_aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import numpy as np


def aperture_photometry(data, coords, radii):
    """
    Perform aperture photometry for a set of coordinates and radii usin photutils.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    coords : np.ndarray
        Array of (x, y) coordinates.
    radii : array-like
        List of aperture radii.

    Returns
    -------
    np.ndarray
        Array of aperture fluxes.
    """
    apertures = [CircularAperture(coords, r=r) for r in radii]
    aperture_fluxes = np.array(
        [photutils_aperture_photometry(data, a)["aperture_sum"].data for a in apertures]
    ).T
    return aperture_fluxes


def annulus_sigma_clip_median(data, coords, r_in, r_out, sigma=3):
    """
    Compute the sigma-clipped median background in an annulus around each coordinate 
    using photutils.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    coords : np.ndarray
        Array of (x, y) coordinates.
    r_in : float
        Inner radius of the annulus.
    r_out : float
        Outer radius of the annulus.
    sigma : float, optional
        Sigma for sigma-clipping.

    Returns
    -------
    np.ndarray
        Array of median background values for each coordinate.
    """
    annulus = CircularAnnulus(coords, r_in, r_out)
    annulus_masks = annulus.to_mask(method="center")

    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        if annulus_data is not None:
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d, sigma=sigma)
            bkg_median.append(median_sigma_clip)
        else:
            bkg_median.append(0.0)

    return np.array(bkg_median)
