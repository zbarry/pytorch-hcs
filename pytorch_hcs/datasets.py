"""
Classes for working with PyTorch Dataloaders and PyTorch-Lightning DataModules
"""


from typing import Callable, Dict

import janitor
import numpy as np
import pytorch_lightning as pl
from pybbbc import BBBC021
from torch.utils.data import DataLoader, Dataset

from .transforms import aug_transform, transform


class BBBC021Dataset(Dataset):
    """
    Attributes:
        moa_df: DatadFrame listing the unique experimental conditions in the
            screen.
        image_df: DataFrame where each row contains all the metadata for a
            corresponding image in the BBBC021 dataset (or subset given
            `image_idcs` you have init'd with).
    """

    def __init__(
        self,
        image_idcs: np.ndarray = None,
        transform: Callable = None,  # pylint: disable=redefined-outer-name
    ):
        """
        Args:
            image_idcs: BBBC021 image indices to include in this dataset.
                Default `None` includes all.
            transform: image transformation / augmentation function that
                takes a NumPy array and returns a PyTorch tensor.
        """

        self.bbbc021 = BBBC021()

        if image_idcs is None:
            image_idcs = np.arange(len(self.bbbc021))

        self.image_idcs = image_idcs

        self.image_df = self.bbbc021.image_df.query(
            "image_idx in @image_idcs"
        ).reset_index(drop=True)

        self.moa_df = self.image_df[
            ["compound", "concentration", "moa", "plate"]
        ].drop_duplicates()

        self.transform = transform

    @property
    def num_classes(self) -> int:
        moas = self.moa_df["moa"].unique()
        return len(moas[moas != "null"])

    @property
    def class_to_label(self) -> Dict[str, int]:
        """
        Return a dictionary mapping the integer class index to the
        human-readable class name as a string.

        NOTE: Depending on train/val/test split, not all classes might be
            represented.
        """
        moas = self.moa_df.sort_values("moa")["moa"].unique()
        moas = moas[moas != "null"]

        mapping = {moa: idx for idx, moa in enumerate(moas)}

        mapping["null"] = -1

        return mapping

    @property
    def label_to_class(self) -> Dict[int, str]:
        """
        See `class_to_label`.
        """
        return {v: k for k, v in self.class_to_label.items()}

    def __len__(self):
        return len(self.image_idcs)

    def __getitem__(self, idx):
        """
        Read the FoV index to fetch from the image stack from the metadata.
        Return the image and its corresponding class label {0...num_classes-1}.
        """

        image_idx = self.image_idcs[idx]

        image, metadata = self.bbbc021[image_idx]

        label = self.class_to_label[metadata.compound.moa]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, metadata

    def compute_class_weights(self) -> np.ndarray:
        """
        Compute class weight for each MoA according to the formula:

        `weight for a class = number of images total / (number of classes * number of images in class)`
        """
        class_counts = (
            self.image_df.query("image_idx in @self.image_idcs")
            .groupby("moa")["image_idx"]
            .count()
            .to_dict()
        )
        class_counts.pop("null")

        # guarantee class order is preserved

        class_counts = {
            class_name: class_counts[class_name]
            for class_name in self.class_to_label.keys()
            if class_name != "null"
        }

        class_counts_arr = np.array(list(class_counts.values()))

        num_fovs = class_counts_arr.sum()

        class_weights = num_fovs / (self.num_classes * class_counts_arr)

        # classes not in this split would have been set to inf

        class_weights[np.isinf(class_weights)] = 0

        return class_weights


class BBBC021DataModule(pl.LightningDataModule):
    """
    PyTorch-Lightning `LightningDataModule` for handling train/val/test split
    of BBBC021 dataset and creating dataloaders.
    """

    def __init__(
        self,
        *,
        num_workers: int = 8,
        tv_batch_size: int = 4,
        t_batch_size: int = 16,
        pin_memory: bool = False,
        train_transform: Callable = aug_transform,
        val_transform: Callable = transform,
        test_transform: Callable = transform,
    ):

        """

        Args:
            num_workers: Number of CPU cores to use for data loading /
                augmentation.
            tv_batch_size: Number of images in a training or validation batch.
            t_batch_size: Number of images in a test batch.
            pin_memory: If using CUDA/GPU, set to `True` for speed improvements
                (False otherise).
            train_transform: Callable function to convert NumPy array to
                PyTorch tensor for training images.
            val_transform: Callable function to convert NumPy array to PyTorch
                tensor for validation images.
            test_transform: Callable function to convert NumPy array to PyTorch
                tensor for test images.
        """

        super().__init__()

        self.num_workers = num_workers
        self.tv_batch_size = tv_batch_size
        self.t_batch_size = t_batch_size
        self.pin_memory = pin_memory

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        # configured in setup()

        self.train_dataset: BBBC021Dataset = None  # type: ignore
        self.val_dataset: BBBC021Dataset = None  # type: ignore
        self.test_dataset: BBBC021Dataset = None  # type: ignore
        self.all_dataset: BBBC021Dataset = None  # type: ignore
        self.all_moa_dataset: BBBC021Dataset = None  # type: ignore

    @staticmethod
    def split(
        dmso_train_frac=0.5, dmso_val_frac=0.35
    ) -> Dict[str, np.ndarray]:
        """
        Splits BBBC021 dataset such that:
        - No plate from DMSO (control) group is used in more than one of the
            train/val/test splits
        - Each compound is only used in one of the splits
        - Train/val/test splits get one compound each from a given MoA and
            if there is a leftover, train split gets the extra.
            There are two MoAs with only two compounds and therefore those
            MoAs are only split into train/val.
        """
        bbbc021 = BBBC021(moa=[moa for moa in BBBC021.MOA if moa != "null"])

        image_df = bbbc021.image_df.query("image_idx not in @BAD_IMAGE_IDCS")

        # split dmso along plate boundaries

        plates = image_df["plate"].unique()
        num_plates = len(plates)

        cdf = np.linspace(0, 1, num_plates)

        train_stop_idx = np.flatnonzero(cdf <= dmso_train_frac)[-1] + 1
        val_stop_idx = (
            np.flatnonzero(cdf <= dmso_train_frac + dmso_val_frac)[-1] + 1
        )

        train_plates = plates[:train_stop_idx]  # noqa: F841
        val_plates = plates[train_stop_idx:val_stop_idx]  # noqa: F841
        test_plates = plates[val_stop_idx:]  # noqa: F841

        train_dmso_idcs = image_df.query(
            'plate in @train_plates and compound == "DMSO"'
        )["image_idx"].values
        val_dmso_idcs = image_df.query(
            'plate in @val_plates and compound == "DMSO"'
        )["image_idx"].values
        test_dmso_idcs = image_df.query(
            'plate in @test_plates and compound == "DMSO"'
        )["image_idx"].values

        # split compounds

        cmpd_im_count_df = (
            image_df.query('compound != "DMSO"')
            .groupby(["compound", "moa"])["site"]
            .count()
            .to_frame("num_images")
            .query("num_images > 0")
            .reset_index()
            .sort_values(["moa", "num_images"], ascending=[True, False])
        )

        train_compounds = []
        val_compounds = []
        test_compounds = []

        for _, cur_moa_df in cmpd_im_count_df.groupby("moa"):
            if len(cur_moa_df) == 0:
                continue

            train_compounds.append(cur_moa_df.iloc[0]["compound"])
            val_compounds.append(cur_moa_df.iloc[1]["compound"])

            try:
                test_compounds.append(cur_moa_df.iloc[2]["compound"])
                train_compounds.append(cur_moa_df.iloc[3]["compound"])
            except IndexError:
                pass

        def fetch_compound_idcs(
            compounds,  # pylint: disable=unused-argument
        ) -> np.ndarray:
            return image_df.query("compound in @compounds")["image_idx"].values

        train_compound_idcs = fetch_compound_idcs(train_compounds)
        val_compound_idcs = fetch_compound_idcs(val_compounds)
        test_compound_idcs = fetch_compound_idcs(test_compounds)

        # merge dmso and compound idcs

        train_idcs = np.concatenate((train_compound_idcs, train_dmso_idcs))
        train_idcs.sort()

        val_idcs = np.concatenate((val_compound_idcs, val_dmso_idcs))
        val_idcs.sort()

        test_idcs = np.concatenate((test_compound_idcs, test_dmso_idcs))
        test_idcs.sort()

        return dict(train=train_idcs, val=val_idcs, test=test_idcs)

    def setup(self, stage=None):
        split = self.split()

        self.train_dataset = BBBC021Dataset(
            split["train"], transform=self.train_transform,
        )

        self.val_dataset = BBBC021Dataset(
            split["val"], transform=self.val_transform,
        )

        self.test_dataset = BBBC021Dataset(
            split["test"], transform=self.test_transform,
        )

        self.all_dataset = BBBC021Dataset(transform=self.test_transform,)

        bbbc021 = BBBC021(moa=[moa for moa in BBBC021.MOA if moa != "null"])

        self.all_moa_dataset = BBBC021Dataset(
            image_idcs=bbbc021.index_vector, transform=self.test_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.tv_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.tv_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.t_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )

    def all_dataloader(self, with_null_moa: bool = True) -> DataLoader:
        """
        Args:
            with_null_moa: If true, images from compounds of unknown MoA are
                included in the dataset.
        """
        dataset = self.all_dataset if with_null_moa else self.all_moa_dataset

        return DataLoader(
            dataset,
            batch_size=self.t_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        return self.all_dataset.num_classes

    @property
    def label_to_class(self) -> Dict[int, str]:
        return self.all_dataset.label_to_class

    @property
    def class_to_label(self) -> Dict[str, int]:
        return self.all_dataset.class_to_label


BAD_IMAGE_IDCS = [
    229,
    523,
    643,
    703,
    1369,
    1429,
    1599,
    1783,
    1903,
    2637,
    2811,
    2875,
    2922,
    2923,
    3049,
    3109,
    3170,
    3352,
    3417,
    3456,
    3576,
    3714,
    3886,
    3887,
    3946,
    3947,
    4006,
    4007,
    4066,
    4067,
    4242,
    4431,
    4607,
    4667,
    4670,
    4727,
    4729,
    4787,
    4977,
    5151,
    5328,
    5388,
    5448,
    5508,
    5697,
    5806,
    6114,
    6174,
    6644,
    6704,
    6826,
    7127,
    7488,
    7489,
    7494,
    7548,
    7549,
    7554,
    7608,
    7609,
    7614,
    7668,
    7669,
    7674,
    7728,
    7729,
    7730,
    7733,
    7734,
    7788,
    7789,
    7790,
    7794,
    7848,
    7849,
    7850,
    7853,
    7854,
    7908,
    7909,
    7910,
    7914,
    8096,
    8271,
    8459,
    8937,
    8997,
    9350,
    9955,
    10073,
    10245,
    10488,
    10594,
    10600,
    10846,
    10859,
    11148,
    11266,
    11574,
    11634,
    11694,
    11754,
    11809,
    11810,
    11863,
    11869,
    11870,
    11871,
    11930,
    11931,
    11990,
    11991,
    12596,
    12650,
    12709,
    12948,
]
