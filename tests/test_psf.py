import numpy as np
from eloy import psf


def test_fit_gaussian_sigma_with_noise():
    rng = np.random.default_rng(12345)
    size = 41
    xs, ys = np.indices((size, size))

    amplitude = 1.0
    background = 0.1
    x0 = 20.3
    y0 = 19.7
    sigma_x_true = 3.0
    sigma_y_true = 2.0
    theta = 0.3

    img = psf.gaussian(
        xs=xs,
        ys=ys,
        amplitude=amplitude,
        x=x0,
        y=y0,
        sigma_x=sigma_x_true,
        sigma_y=sigma_y_true,
        theta=theta,
        background=background,
    )

    noise = rng.normal(scale=0.02, size=img.shape)
    data = img + noise

    res = psf.fit_gaussian(data)

    # fit_gaussian enforces sigma_x >= sigma_y; compare sorted values
    fitted = sorted([res["sigma_x"], res["sigma_y"]], reverse=True)
    true = sorted([sigma_x_true, sigma_y_true], reverse=True)

    rel_err_x = abs(fitted[0] - true[0]) / true[0]
    rel_err_y = abs(fitted[1] - true[1]) / true[1]

    # require both sigmas to be recovered within 10% relative error
    assert rel_err_x < 0.10 and rel_err_y < 0.10
