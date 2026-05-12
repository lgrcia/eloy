from eloy import utils, centroid, alignment
from photutils.centroids import centroid_com
import numpy as np
import pytest


def test_centroid():
    data, coords = utils.fake_image()
    centroids = centroid.photutils_centroid(
        data, coords, cutout=3, centroid_fun=centroid_com
    )
    match = alignment.count_cross_match(centroids, coords, tol=1)
    assert match > len(coords) - 2


def test_centroid_out():
    """photutils centroiding check that sources are in images. As a feature we only
    centroid sources that are in"""

    data = np.random.rand(50, 50)
    coords = np.array([[-1, 1], [20, 20]])
    centroid.photutils_centroid(data, coords)


def test_ballet_nans():
    """Test ballet centroid with nans"""
    pytest.importorskip("jax")
    from eloy.centroid import ballet_centroid, Ballet

    cnn = Ballet()

    im = np.random.rand(20, 20)
    coords = np.array([[10.0, 10.0], [-20.0, -20.0]])

    assert not np.any(np.isnan(ballet_centroid(im, coords, cnn, nans=False)))
    assert np.any(np.isnan(ballet_centroid(im, coords, cnn, nans=True)))


def test_ballet_centroid_location():
    """Test that the ballet centroid is close to the input coordinates"""
    pytest.importorskip("jax")
    from eloy.ballet.training import Moffat2D
    from eloy.centroid import ballet_centroid, Ballet

    rng = np.random.default_rng(0)
    truth = [20.4, 19.6]
    moffat2d = Moffat2D(cutout_size=41)
    # Generate a synthetic star with Moffat profile and gaussian noise
    # eloy's Moffat2D uses np.indices, so x0 is the row index and y0 is the column
    synth_star = moffat2d.moffat2D_model(
        1000.0,
        truth[1], truth[0], # x0, y0 (reversal because of row/column indexing)
        2.0, 2.0,           # sx, sy
        0, 0,               # theta, b
        3.0                 # beta
    )
    synth_star = synth_star + 50.0 + rng.normal(0.0, 2.0, size=synth_star.shape)
    cnn = Ballet()
    centroid_estimate = ballet_centroid(synth_star, np.asarray([truth]), cnn, nans=True)
    assert np.allclose(centroid_estimate, np.array([truth]), atol=0.01)
