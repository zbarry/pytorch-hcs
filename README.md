# pytorch-hcs

Prediction of drug mechanism-of-action (MoA) from high content screening images
using convolutional neural networks.

## Background

### BBBC021 dataset

See the [BBBC021 landing page](https://bbbc.broadinstitute.org/BBBC021) for more info on the dataset.

<img src="https://data.broadinstitute.org/bbbc/BBBC021/aurora-kinase-inhibitor.png" width="250" />
<img src="https://data.broadinstitute.org/bbbc/BBBC021/tubulin-stabilizer.png" width="250" />
<img src="https://data.broadinstitute.org/bbbc/BBBC021/monoaster.png" width="250" />

Example images from BBBC021.

### The task

- Given a multi-channel fluorescence image of MCF-7 cells,
predict the mechanism-of-action of the compound the cells were treated with.

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

- PyTorch and PyTorch-Lightning - PTL reduces training boilerplate.
- Weights and Biases - stores our training runs and model checkpoints.

### Python package

Reusable code modules are found in the `pytorch_hcs` package.

* `datasets.py` - PyTorch dataset and PyTorch-Lightning DataModule for working with BBBC021 data.
* `models.py` - PyTorch-Lightning modules wrapping CNN models.
* `transforms.py` - image transforms for data augmentation.

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

- Decrease plate effects on embeddings (e.g., through adversarial learning).
- Add hyperparameter sweep capability using Weights and Biases / improve model classification performance.
- Log model test set evaluation results to W&B.
- Move BBBC021 dataset to [ActiveLoop Hub](https://docs.activeloop.ai/) to speed up download / dataset prep times.
