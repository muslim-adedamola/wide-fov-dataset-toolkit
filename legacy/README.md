# Legacy RMFV365 Scripts

This folder contains the original scripts used to generate the image transformations and YOLO-format bounding-box annotations for RMFV365.

The scripts are preserved for reproducibility and historical traceability. They are not yet the cleaned toolkit interface.

## Structure

- `coordinates/`: scripts for transforming object bounding-box coordinates and writing YOLO label files.
- `transformations/`: scripts for transforming the corresponding images.

## Original workflow

The original RMFV365 generation pipeline was split into two parallel steps:

1. Transform the original Objects365 images using a selected wide-FoV/fisheye transformation.
2. Apply the matching coordinate transformation to each bounding box and export the result in YOLO format.

For example:

- `transformation_7.py` applies the camera/lens-independent fisheye transformation with `n = 7`.
- `coordinates_7.py` applies the corresponding bounding-box coordinate transformation.
- `transformation_barrel2.py` applies the barrel distortion variant.
- `coordinates_barrel2.py` applies the corresponding barrel bounding-box transformation.
- `transformation_div2.py` applies the division-model distortion variant.
- `coordinates_div2.py` applies the corresponding division-model bounding-box transformation.

## Notes

These scripts contain hardcoded paths, dataset names, patch names, and output directories from the original RMFV365 experiments. The cleaned version of this repository will refactor this logic into reusable modules and command-line tools.
