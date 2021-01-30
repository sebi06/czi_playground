# czitools

## Description

Collection of functions to read, analyze, display and segment CZI (and OME-TIFF) images.

- **imgfile_tools** - collection to functions to read the image metadata and also display those inside jupyter notebooks or using the Napari viewer

- **segmentation_tools** - collection of function to segment images using

  - scikit-image
  - pretrained networks used in ZEN
  - StarDist2D segmentation
  - CellPose segmentation

- **czifile_tools** - collection of CZI-specific tools

- **visu_tools** - collection of tools to display and visualize results, e.g. heatmaps
***
## Important Remarks

When using this package to segment images using StarDist, CellPose make sure that the dependencies for those are fulfilled. Those will be not automatically installed yb this package.
***
## Installation

```bash
cd your_project

# Download the setup.py file:
#  download with wget
wget https://github.com/sebi06/czitools/blob/master/setup.py -O setup.py

#  download with curl
curl -O https://github.com/sebi06/czitools/blob/master/setup.py
```
***
## Disclaimer

This package is purely experimental. Use it at your own risk.
