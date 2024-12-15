from photutils.aperture import aperture_photometry as photutils_aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import numpy as np


def aperture_photometry(data, coords, radii):
    apertures = [CircularAperture(coords, r=r) for r in radii]
    aperture_fluxes = np.array(
        [photutils_aperture_photometry(data, a)["aperture_sum"].data for a in apertures]
    ).T
    return aperture_fluxes


def annulus_sigma_clip_median(data, coords, r_in, r_out, sigma=3):
    annulus = CircularAnnulus(coords, r_in, r_out)
    annulus_masks = annulus.to_mask(method="center")

    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        if annulus_data is not None:
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d, sigma=sigma)
            bkg_median.append(median_sigma_clip)
        else:
            bkg_median.append(0.0)

    return np.array(bkg_median)
