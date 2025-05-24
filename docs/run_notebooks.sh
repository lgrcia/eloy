#!/bin/sh

rm -r docs/calibrated_images
jupyter-execute --inplace docs/calibration.ipynb
jupyter-execute --inplace docs/stacking.ipynb
jupyter-execute --inplace docs/plate-solving.ipynb
jupyter-execute --inplace docs/video.ipynb
jupyter-execute --inplace docs/photometry.ipynb
