from astropy.nddata import Cutout2D
import numpy as np
from astropy.nddata import Cutout2D


def cutout(data, coords, shape, wcs=None):
    values = []
    for coords in coords:
        cutout = Cutout2D(
            data, coords, shape, wcs=wcs, fill_value=np.nan, mode="partial"
        )
        values.append(cutout.data)
    return np.array(values)


def cutout(data, coords, shape, wcs=None):
    values = []
    for coords in coords:
        cutout = Cutout2D(
            data, coords, shape, wcs=wcs, fill_value=np.nan, mode="partial"
        )
        values.append(cutout.data)
    return np.array(values)


def std_diff_metric(fluxes):
    k = len(list(np.shape(fluxes)))
    return np.std(np.diff(fluxes, axis=k - 1), axis=k - 1)


def stability_aperture(fluxes):
    lc_c = np.abs(np.diff(fluxes, axis=0))
    return np.mean(lc_c, axis=1)


def index_binning(x, size):
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
