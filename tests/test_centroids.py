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
