# pytorch-hcs

Convolutional neural network-based prediction of drug mechanism-of-action (MoA) from high content screening images
and use of CNN image embeddings to find outliers/novel images.

## Background

### High content screening / imaging

Fluorescence microscopy is a core tool in biological and drug discovery.
High content screening automates fluorescence microscopy on a mass scale,
allowing researchers to understand the impact of thousands of perturbations
on cellular morphology and health in a single assay.
Screens specifically focused on treatment of cells with biologically active molecules / drugs
can lend insight into the function of those compounds based on how they modulate the imaged cellular structures.
Functional insights can lead to identification of compound "hits" for potential drug candidates.

### BBBC021 dataset

See the [BBBC021 landing page](https://bbbc.broadinstitute.org/BBBC021) for more info on the dataset.

tl;dr:

- Human breast cancer cell line (MCF-7) treated with various compounds of both known and unknown MoA.
- Following treatment, cells are stained for their nuclei (blue) and the cytoskeletal proteins tubulin (green) and actin (red).

| Aurora kinase inhibitor | Tubulin stabilizer | Eg5 inhibitor |
|-------------------------|--------------------|---------------|
|<img src="https://data.broadinstitute.org/bbbc/BBBC021/aurora-kinase-inhibitor.png" width="200" />|<img src="https://data.broadinstitute.org/bbbc/BBBC021/tubulin-stabilizer.png" width="200" /> | <img src="https://data.broadinstitute.org/bbbc/BBBC021/monoaster.png" width="200" /> |

Example images from BBBC021.

### Project goals

- Given a multi-channel fluorescence image of MCF-7 cells,
train a convolutional neural network to predict the mechanism-of-action of the compound the cells were treated with.
- Use the trained CNN to extract image embeddings.
- Perform UMAP dimensionality reduction on embeddings for dataset visualization and exploration.
- Find interesting / artifactual image outliers in the BBBC021 dataset using image embeddings.

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

- PyTorch and [PyTorch-Lightning](https://www.pytorchlightning.ai/) - PTL reduces training boilerplate.
- [Weights and Biases](wandb.ai) - stores training runs and model checkpoints.

### Python package

Reusable code modules are found in the `pytorch_hcs` package.

* `datasets.py` - PyTorch dataset and PyTorch-Lightning DataModule for working with BBBC021 data.
* `models.py` - PyTorch-Lightning modules wrapping CNN models.
* `transforms.py` - image transforms for data augmentation.

### Notebooks

The code that orchestrates the modules found in the Python package is in notebooks in the `notebooks/` folder.

#### Available notebooks (by order of execution):

1. `01_download_bbbc021.ipynb` - download raw BBBC021 images and pre-process them using pybbbc.
2. `02_bbbc021_visualization.ipynb` - explore the BBBC021 dataset with an interactive visualization.
3. `03_train_model.ipynb` - train a CNN to predict MoA from BBBC021 images.
4. `04_evaluate_model.ipynb` - evaluate performance of trained CNN on test set.
5. `05_visualize_embeddings.ipynb` - produce image embeddings, UMAP them, visualize and find outliers.

#### Extras:

* `dataset_cleaning_visualization.ipynb` - manually step through BBBC021 with a visualization
to label images in the training and validation sets as "good" or "bad".
* `notebooks/analysis/umap_param_sweep.ipynb` - sweep through UMAP parameterizations to assess impact on resulting embeddings.

## Development

### Install pre-commit hooks

These will clear notebook outputs as well as run code formatters upon commit.

`pre-commit install`

### Ways to contribute

- Decrease plate effects on embeddings (e.g., through adversarial learning).
- Add hyperparameter sweep capability using Weights and Biases / improve model classification performance.
- Log model test set evaluation results to W&B.
- Make better use of W&B in general for tracking results.
- Move BBBC021 dataset to [ActiveLoop Hub](https://docs.activeloop.ai/) to speed up download / dataset prep times.
