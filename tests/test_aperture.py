import numpy as np
import pytest

from eloy.photometry import aperture_photometry


@pytest.mark.parametrize('mask', [None, True])
def test_aperture_photometry(mask):
    # Create a synthetic image with a bright spot
    image = np.zeros((100, 100))
    expected_value = 1000

    image[50, 50] = expected_value  # Bright spot at the center

    if mask is not None:
        # Create a mask that excludes the bright spot
        mask = np.zeros_like(image, dtype=bool)
        mask[50, 50] = True  # Mask out the bright spot
        expected_value = 0  # Expect zero flux when the bright spot is masked

    # Define the aperture radius
    radii = [5, 10]

    # Perform aperture photometry
    flux = aperture_photometry(image, (50, 50), radii, mask=mask)

    # The expected flux should be close to the value of the bright spot
    assert np.isclose(flux, expected_value, atol=1e-2).all(), f"Expected flux to be close to {expected_value}, got {flux}"
