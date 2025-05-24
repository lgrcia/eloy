import numpy as np
from astropy.io import fits


def divisors(n):
    _divisors = []
    i = 1
    while i <= n:
        if n % i == 0:
            _divisors.append(i)
        i = i + 1
    return np.array(_divisors)


def easy_median(images, m=50):
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
    with np.errstate(divide="ignore", invalid="ignore"):
        calibrated_data = (data - (dark * exposure + bias)) / flat

    calibrated_data[calibrated_data < 0] = np.nan
    calibrated_data[~np.isfinite(calibrated_data)] = -1
    return calibrated_data
