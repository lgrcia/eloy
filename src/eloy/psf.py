from scipy.optimize import minimize
import numpy as np

gaussian_sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))


def moments(data):
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


def fit_gaussian(data, init=None):
    x, y = np.indices(data.shape)

    def model(height, xo, yo, sx, sy, theta, m):
        dx = x - xo
        dy = y - yo
        a = (np.cos(theta) ** 2) / (2 * sx**2) + (np.sin(theta) ** 2) / (2 * sy**2)
        b = -(np.sin(2 * theta)) / (4 * sx**2) + (np.sin(2 * theta)) / (4 * sy**2)
        c = (np.sin(theta) ** 2) / (2 * sx**2) + (np.cos(theta) ** 2) / (2 * sy**2)
        psf = height * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))
        return psf + m

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
    return {k: v for k, v in zip(keys, opt)}
