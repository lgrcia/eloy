"""
Differential photometry and flux optimization routines.

This module provides functions for computing differential fluxes,
optimizing comparison star weights, and selecting optimal flux indices.
"""

import numpy as np
from eloy import utils


def weights(
    fluxes: np.ndarray, tolerance: float = 1e-3, max_iteration: int = 200, bins: int = 5
):
    """
    Returns the weights computed using Broeg 2005.

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes matrix with dimensions (star, flux) or (aperture, star, flux).
    tolerance : float, optional
        Minimum standard deviation of weights difference to attain (weights are stable).
    max_iteration : int, optional
        Maximum number of iterations to compute weights.
    bins : int, optional
        Binning size (in number of points) to compute the white noise.

    Returns
    -------
    np.ndarray
        Broeg weights.
    """

    # normalize
    dfluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)

    def weight_function(fluxes):
        return 1 / np.std(fluxes, axis=-1)

    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros(dfluxes.shape[0 : len(dfluxes.shape) - 1])

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = weight_function(dfluxes)
            mask = np.where(~np.isfinite(weights))
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            weights = weight_function(lcs)

        weights[~np.isfinite(weights)] = 0

        evolution = np.abs(
            np.nanmean(weights, axis=-1) - np.nanmean(last_weights, axis=-1)
        )

        last_weights = weights
        lcs = diff(dfluxes, weights=weights)

        i += 1

    weights[0, mask] = 0

    return weights[0]


def diff(fluxes: np.ndarray, weights: np.ndarray = None):
    """
    Returns differential fluxes.

    If weights are specified, they are used to produce an artificial light curve by which all flux are differentiated (see Broeg 2005).

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes matrix with dimensions (star, flux) or (aperture, star, flux).
    weights : np.ndarray, optional
        Weights matrix with dimensions (star) or (aperture, star).

    Returns
    -------
    np.ndarray
        Differential fluxes if weights is provided, else normalized fluxes.
    """
    diff_fluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)
    if weights is not None:
        # not to divide flux by itself
        sub = np.expand_dims((~np.eye(fluxes.shape[-2]).astype(bool)).astype(int), 0)
        weighted_fluxes = diff_fluxes * np.expand_dims(weights, -1)
        # see broeg 2005
        artificial_light_curve = (sub @ weighted_fluxes) / np.expand_dims(
            weights @ sub[0], -1
        )
        diff_fluxes = diff_fluxes / artificial_light_curve
    return diff_fluxes


def auto_diff_1d(fluxes, i=None):
    """
    Automatically compute differential fluxes and optimal weights for 1D flux array.

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes array.
    i : int or None, optional
        Index of the target star.

    Returns
    -------
    tuple
        Tuple (differential fluxes, weights).
    """
    dfluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)
    w = weights(dfluxes)
    if i is not None:
        idxs = np.argsort(w)[::-1]
        white_noise = utils.binned_nanstd(dfluxes)
        last_white_noise = 1e10

        def best_weights(j):
            _w = w.copy()
            _w[idxs[j::]] = 0.0
            _w[i] = 0.0
            return _w

        for j in range(w.shape[-1]):
            _w = best_weights(j)
            _df = diff(dfluxes, _w)
            _white_noise = np.take(white_noise(_df), i, axis=-1)[0]
            if not np.isfinite(_white_noise):
                continue
            if _white_noise < last_white_noise:
                last_white_noise = _white_noise
            else:
                break

        w = best_weights(j - 1)

    df = diff(dfluxes, w)

    return df.reshape(fluxes.shape), w


def auto_diff(fluxes: np.array, i: int = None):
    """
    Automatically compute differential fluxes and optimal weights for 2D or 3D flux array.

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes array.
    i : int or None, optional
        Index of the target star.

    Returns
    -------
    tuple or tuple of arrays
        Differential fluxes and weights.
    """
    if fluxes.ndim == 3:
        auto_diffs = [auto_diff_1d(f, i) for f in fluxes]
        w = [a[1] for a in auto_diffs]
        fluxes = np.array([a[0] for a in auto_diffs])
        return fluxes, np.array(w)
    else:
        return auto_diff_1d(fluxes, i)


def optimal_flux(diff_fluxes, method="stddiff", sigma=4):
    """
    Select the optimal flux index based on a given criterion.

    Parameters
    ----------
    diff_fluxes : np.ndarray
        Differential fluxes array.
    method : str, optional
        Criterion method: "binned", "stddiff", or "stability".
    sigma : float, optional
        Sigma clipping threshold.

    Returns
    -------
    int
        Index of the optimal flux.
    """
    fluxes = diff_fluxes.copy()
    fluxes = fluxes[
        ...,
        np.all(
            (fluxes - np.median(fluxes, 1)[..., None])
            < sigma * np.std(fluxes, 1)[..., None],
            0,
        ),
    ]
    if method == "binned":
        white_noise = utils.binned_nanstd(fluxes)
        criterion = white_noise(fluxes)
    elif method == "stddiff":
        criterion = utils.std_diff_metric(fluxes)
    elif method == "stability":
        criterion = utils.stability_aperture(fluxes)
    else:
        raise ValueError("{} is not a valid method".format(method))

    i = np.argmin(criterion)
    return i
