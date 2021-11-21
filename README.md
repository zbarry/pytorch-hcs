# pytorch-hcs

Prediction of drug mechanism-of-action (MoA) from high content screening images.

## Background

![](https://data.broadinstitute.org/bbbc/BBBC021/aurora-kinase-inhibitor.png)

Example image from [BBBC021](https://bbbc.broadinstitute.org/BBBC021) dataset
used in this work.

## Getting started

1. Clone the repository:

```
git clone https://github.com/zbarry/pytorch-hcs.git
```

2. Install the environment and the `pytorch_hcs` package:

```
cd pytorch-hcs
conda install -c conda-forge mamba -y
mamba env update
```

(the `mamba` install is optional but recommended as a conda replacement which has much faster dependency solves)

This will create a `pytorch-hcs` environment and `pip install` the Python package in one go.

A [fork](https://github.com/zbarry/pybbbc) of [pybbbc](https://github.com/giacomodeodato/pybbbc) will also be installed.
We use this to download the BBBC021 dataset and access individual images and metadata.

3. Acquire the BBBC021 dataset

Either run `notebooks/download_bbbc021.ipynb` from top to bottom or in a Python terminal (with the `pytorch-hcs` environment activated):

```python
from pybbbc import BBBC021

BBBC021.download()
BBBC021.make_dataset(max_workers=2)

# test
bbbc021 = BBBC021()
bbbc021[0]
```

## Project structure

### Python package

Reusable code modules are found in the `pytorch_hcs` package. 

### Notebooks

The code that orchestrates the modules found in the Python package is in notebooks in the `notebooks/` folder.

Notebook order of execution:

1. download_bbbc021.ipynb
2. bbbc021_visualization.ipynb
3. train_model.ipynb
4. evaluate_model.ipynb
5. umap_param_sweep.ipynb

Extras:
* dataset_cleaning_visualization.ipynb
