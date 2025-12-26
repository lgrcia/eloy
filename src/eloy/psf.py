"""
Point Spread Function (PSF) modeling utilities.

This module provides functions for estimating PSF parameters using image moments
and fitting 2D Gaussian models to image data.
"""

from scipy.optimize import minimize
from functools import partial
import numpy as np

gaussian_sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))


def moments(data):
    """
    Estimate the moments of a 2D distribution.

    Parameters
    ----------
    data : np.ndarray
        2D image data.

    Returns
    -------
    dict
        Dictionary of estimated parameters: amplitude, x, y, sigma_x, sigma_y, background, theta, beta.
    """
    height = data.max()
    background = data.min()
    data = data - np.min(data)
    total = data.sum()
    x, y = np.indices(data.shape)
    x = (x * data).sum() / total
    y = (y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width_x /= gaussian_sigma_to_fwhm
    width_y /= gaussian_sigma_to_fwhm
    return {
        "amplitude": height,
        "x": x,
        "y": y,
        "sigma_x": width_x,
        "sigma_y": width_y,
        "background": background,
        "theta": 0.0,
        "beta": 3.0,
    }


def gaussian(
    xs=None,
    ys=None,
    amplitude=None,
    x=None,
    y=None,
    sigma_x=None,
    sigma_y=None,
    theta=None,
    background=None,
    **kwargs,
):
    dx = xs - x
    dy = ys - y
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    psf = amplitude * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))
    return psf + background


def fit_gaussian(data, init=None):
    """
    Fit a 2D Gaussian to the data.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    init : dict or None, optional
        Initial parameter estimates.

    Returns
    -------
    dict
        Dictionary of fitted parameters: amplitude, x, y, sigma_x, sigma_y, theta, background.
    """
    x, y = np.indices(data.shape)

    model = partial(gaussian, x, y)

    def nll(params):
        ll = np.sum(np.power(model(*params) - data, 2))
        return ll

    keys = ["amplitude", "x", "y", "sigma_x", "sigma_y", "theta", "background"]
    if init is None:
        p0 = moments(data)
    else:
        p0 = init

    p0 = [p0[k] for k in keys]
    w = np.max(data.shape)
    bounds = [
        (0, 1.5),
        *((0, w),) * 2,
        *((0.5, w),) * 2,
        (-np.pi, np.pi),
        (0, np.mean(data)),
    ]

    opt = minimize(nll, p0, bounds=bounds).x
    result = {k: v for k, v in zip(keys, opt)}

    # Ensure sigma_x is the larger value and sigma_y is the smaller
    if result["sigma_x"] < result["sigma_y"]:
        result["sigma_x"], result["sigma_y"] = result["sigma_y"], result["sigma_x"]
        result["theta"] += np.pi / 2

    # Ensure theta is always positive
    result["theta"] = result["theta"] % (2 * np.pi)

    return result
