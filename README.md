# ND2Handler

`nd2Handler` is a Python utility for processing ZEISS `.nd2` microscopy files. It allows you to batch-extract images across channels, time frames, views, and z-stacks; apply intensity projections; adjust contrast; and save results as JPG or TIFF files.

## Features

- **Flexible selection** of channels, frames, views, and z-layers (per channel) using simple range strings (e.g., "1-3,5").
- **Intensity projection** options: maximum or average along the z-axis.
- **Contrast adjustment** and intensity offsets (per channel each).
- **Batch processing** of entire folders of ND2 files.
- **Output formats**: individual 2D images (`.jpg` - `8bit` | `.tif`, `16bit`), or stacked 3D TIFF.


## Installation

I suggest to use Anaconda for environment management.
1. Please make sure that Anaconda is properly installed ([HowTo](https://www.anaconda.com/docs/getting-started/anaconda/install)).
2. Initialize anaconda for all shells. Run this command in your Anaconda prompt:
  ~~~batch
  conda init --all
  ~~~

3. **On Windows:**\
Run the bundled batch file `conda_virtual_environment_create.bat` per double-click for an automatic setup, reinstalling if the environment exists.\
**On other platforms:**\
Please use your terminal manually:

~~~bash
conda env create -n nd2venv -f environment.yml
conda activate nd2venv
~~~

## Quickstart

A comprehensive interactive quickstart is provided in the `example.ipynb` Jupyter notebook.

This notebook demonstrates:

- How a string input is parsed
- Creating an \`nd2Handler\` instance with custom settings.
- Using \`process_folder\` to batch-export images.
- Inspecting file axes with \`get_axes_info\`.