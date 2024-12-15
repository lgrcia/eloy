from photutils.centroids import (
    centroid_quadratic,
    centroid_sources,
)
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

default_centroid_func = centroid_quadratic


def photutils_centroid(data, coords, cutout=21, centroid_fun=None):
    if centroid_fun is None:
        centroid_fun = default_centroid_func

    x, y = coords.T.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        centroid_coords = np.array(
            centroid_sources(data, x, y, box_size=cutout, centroid_func=centroid_fun)
        ).T
    idxs = np.flatnonzero(~np.all(np.isfinite(centroid_coords), 1))
    centroid_coords[idxs] = coords[idxs]
    return centroid_coords
