from astropy.io import fits
from dateutil import parser
from glob import glob
from collections import defaultdict
from pathlib import Path
from datetime import timedelta
from eloy import calibration

from astropy.io import fits
from dateutil import parser
from glob import glob
from collections import defaultdict
from pathlib import Path
from datetime import timedelta
from eloy import calibration


def load_calibration_files():

    files = glob("./photometry_raw_data/**/*.fit*")
    files_meta = defaultdict(dict)
    observations = defaultdict(lambda: defaultdict(int))

    for file in files:
        header = fits.getheader(file)
        file_date = parser.parse(header["DATE-OBS"])
        # because some observations are taken over midnight
        file_date = file_date - timedelta(hours=10)
        files_meta[file]["date"] = file_date
        files_meta[file]["type"] = Path(file).parent.stem
        observations[file_date.date()][files_meta[file]["type"]] += 1

    # only picking up the science images
    lights = list(filter(lambda f: files_meta[f]["type"] == "ScienceImages", files))
    # sorting them by date
    lights = sorted(lights, key=lambda f: files_meta[f]["date"])
    # selecting the first one
    file = lights[0]

    def filter_files(files, file_type):
        return list(filter(lambda f: files_meta[f]["type"] == file_type, files))

    biases = filter_files(files, "Bias")
    darks = filter_files(files, "Darks")
    flats = filter_files(files, "Flats")

    bias = calibration.master_bias(files=biases)
    dark = calibration.master_dark(files=darks, bias=bias)
    flat = calibration.master_flat(files=flats, bias=bias, dark=dark)

    return bias, dark, flat, lights


def process_image(index_file, shared_data=None):
    i, file = index_file
    image = fits.getdata(file)
    header = fits.getheader(file)

    # applying the master calibration
    calibrated_image = calibration.calibrate(
        image,
        exposure=header["EXPTIME"],
        bias=shared_data["bias"],
        dark=shared_data["dark"],
        flat=shared_data["flat"],
    )

    return i, calibrated_image


if __name__ == "__main__":

    import multiprocessing as mp
    from tqdm import tqdm
    from functools import partial
    from eloy import utils

    bias, dark, flat, lights = load_calibration_files()

    master_files = {
        "bias": bias,
        "dark": dark,
        "flat": flat,
    }

    shared_data = utils.share_data(master_files)
    indexes_images = list(enumerate(lights))
    calibrated_images = {}

    with mp.Pool() as pool:
        for i, calibrated_image in tqdm(
            pool.imap(partial(process_image, shared_data=shared_data), indexes_images),
            total=len(indexes_images),
        ):
            calibrated_images[i] = calibrated_image
