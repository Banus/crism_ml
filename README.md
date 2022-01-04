# CRISM Machine Learning Toolkit

## Introduction

This package demonstrates the utility of machine learning in two important
tasks in hyperspectral image analysis: nonlinear noise removal (*ratioing*) and
mineral classification.

We developed a model to classify spectra in satellite hyperspectral image from
Mars, acquired by the [CRISM](http://crism.jhuapl.edu/) experiment.
Specifically, we implement a *Hierarchical Bayesian Model (HBM)* based on a
Gaussian model for spectra, with a global [Normal-Inverse Wishart](https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution)
prior and a separate local Normal prior for each image.

The HBM is used to identify pixels lacking definite spectral features (i.e.,
bland) and use them divide (ratio) the remaining spectra on an image column to
remove nonlinear noise and distortions.
A second model is trained on ratioed spectra to classify a set of 33 distinct
mineral classes.

The code offers the following functionality:

* Train a HBM to classify bland pixels and minerals

* Preprocessing utilities to clean and ratio the spectra

* Plotting utilities to generate a false-color image, show predictions and
  per-region spectra

## Usage

The code requires Python 3.7 or newer and it runs on Windows and Linux.
You can install an environment with all the dependencies with:

```bash
conda env create -f environment.yml
```

A `requirements.txt` file is also available if you do not have Anaconda; run
this in your virtual environment:

```bash
pip install -r requirements.txt
```

To use the package from any path, install it in your environment with the
following command from the project root:

```bash
pip install -e .
```

This command will also take care of the project dependencies if you didn't
install them and add the project to the python packages in editable mode.

### Running example

To check that everything is working, download an image from the CRIM website
and run the main script on it. The script witll train the models, preprocess
the image and classify the pixels. A set of plots will be saved in the
`workdir/plot` directory (using GIT bash):

```bash
# download dataset
mkdir -p datasets && cd datasets
curl -O http://cs.iupui.edu/~mdundar/CRISM/CRISM_bland_unratioed.mat
curl -O http://cs.iupui.edu/~mdundar/CRISM/CRISM_labeled_pixels_ratioed.mat
cd ..

# download image
curl -O https://pds-geosciences.wustl.edu/mro/mro-m-crism-3-rdr-targeted-v1/mrocr_2104/trdr/2010/2010_056/hrl00016cfe/hrl00016cfe_07_if181l_trr3.img
curl -O https://pds-geosciences.wustl.edu/mro/mro-m-crism-3-rdr-targeted-v1/mrocr_2104/trdr/2010/2010_056/hrl00016cfe/hrl00016cfe_07_if181l_trr3.lbl

python crism_ml/train.py hrl00016cfe_07_if181l_trr3 --plot
```

A detailed guide of all the steps performed by the classification script is
available in `tutorials/Training.ipynb`.

## Dataset

We released two datasets on the
[CRISM ML toolkit](http://cs.iupui.edu/~mdundar/CRISM.htm) website, to train
the bland pixel model and the mineral model.
Download them to the `datasets` directory or pass the path to the`train.py`
script using the `--datapath` argument.

The bland pixel dataset has the following variables:

| **Name**   | **Size**   | **Description**                                    |
| ----       | ----       | -----------                                        |
| `pixspec`  | 337617×350 | Unratioed spectra                                  |
| `im_names` | 340        | List of CRISM image names, mapping to numerical ID |
| `pixims`   | 337617     | Numerical ID of the original image                 |
| `pixcrds`  | 337617×2   | (x,y) point coordinates in the original image      |  

And the mineral dataset has the following structure:

| **Name**   | **Size**   | **Description**                                    |
| ----       | ----       | -----------                                        |
| `pixspec`  | 592413×350 | Ratioed spectra                                    |
| `pixlabs`  | 592413     | Mineral labels                                     |
| `im_names` | 77         | List of CRISM image names, mapping to numerical ID |
| `pixims`   | 592413     | Numerical ID of the original image                 |
| `pixpat`   | 592413     | ID of the connected patch the pixel belongs to     |
| `pixcrds`  | 337617×2   | (x,y) point coordinates in the original image      |

## License

The code is released under the Apache-2 License (see `LICENSE.txt` for
details).

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{Plebani2022crism,
  title = {A machine learning toolkit for {CRISM} image analysis},
  journal = {Icarus},
  pages = {114849},
  year = {2022},
  issn = {0019-1035},
  doi = {https://doi.org/10.1016/j.icarus.2021.114849},
  url = {https://www.sciencedirect.com/science/article/pii/S0019103521004905},
  author = {Emanuele Plebani and Bethany L. Ehlmann and Ellen K. Leask and Valerie K. Fox and M. Murat Dundar},
}
```
