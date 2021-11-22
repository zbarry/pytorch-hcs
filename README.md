# pytorch-hcs

Prediction of drug mechanism-of-action (MoA) from high content screening images.

## Background

![](https://data.broadinstitute.org/bbbc/BBBC021/aurora-kinase-inhibitor.png | width=100)
![](https://data.broadinstitute.org/bbbc/BBBC021/tubulin-stabilizer.png)

<img src="https://data.broadinstitute.org/bbbc/BBBC021/tubulin-stabilizer.png" width="200" />

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

There are a lot of files to download.
Plan on this process taking hours.

## Project structure

### Key dependencies

- PyTorch and PyTorch-Lightning
- Weights and Biases

### Python package

Reusable code modules are found in the `pytorch_hcs` package.

* `datasets.py` - PyTorch dataset and PyTorch-Lightning DataModule for working with BBBC021 data.
* `models.py` - PyTorch-Lightning modules wrapping CNN models.
* `transforms.py` - image transforms for data augmentation.
* `vis.py` - helper functions for visualization with HoloViews.

### Notebooks

The code that orchestrates the modules found in the Python package is in notebooks in the `notebooks/` folder.

#### Notebook order of execution:

1. `01_download_bbbc021.ipynb`
2. `02_bbbc021_visualization.ipynb`
3. `03_train_model.ipynb`
4. `04_evaluate_model.ipynb`
5. `05_visualize_outliers.ipynb`

#### Extras:

* `dataset_cleaning_visualization.ipynb`
* `notebooks/analysis/umap_param_sweep.ipynb`

## Development

### Install pre-commit hooks

`pre-commit install`

### Ways to contribute

- Decrease plate effects on embeddings (e.g., through adversarial learning)
- Add hyperparameter sweep capability using Weights and Biases
- Log model test set evaluation results to W&B
- Move BBBC021 dataset to [ActiveLoop Hub](https://docs.activeloop.ai/) to speed up download / dataset prep times
