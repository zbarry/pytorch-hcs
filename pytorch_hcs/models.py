"""
PyTorch-Lightning-compatible convolutional networks for classification.
"""

# pylint: disable=no-member  # can't parse torch package

import inspect

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchvision.models import resnet18, resnet50, resnet101


class ClassificationModel(pl.LightningModule):
    """
    Base class for classification models that configures optimizers and
    implements generic training / validation / test steps.

    NOTE: Models must implement the keyword arguments in the __init__ docstring

    Subclasses should also implement:
    - `.model` - a PyTorch `nn.Module` whose `.forward()` returns embeddings that
        would feed into the final classification layer.
    - `.model.classifier` - the final classification head layer(s). Should be
        disconnected from the rest of the layers and will be called manually in
        `ClassificationModel.forward()`.
    """

    def __init__(self):
        """
        Implement the following keyword arguments in your subclass
        if you wish to use the default optimizer configuration.
        Otherwise, you can override `configure_optimizers` in your subclass.

        Whatever named keyword arguments your __init__ accepts will be saved to
        the saved checkpoint (.ckpt) file.

        These parameters are accessible via the `.hparams` instance attribute:
        ```
        num_classes = model.hparams.num_classes
        ```

        Args:
            num_classes: Number of classes.
            learning_rate: Optimizer learning rate.
            lambdalr_factor: geometric learning rate decay constant per
                epoch ( lr_{epoch+1} = factor * lr_epoch )
            plateau_patience: How many epochs must pass without validation
                loss improving to be considered a "plateau" and lower the
                learning rate.
            plateau_factor: Factor to multiply current learning rate by
                when plateau condition is reached.
            train_class_weights: Array of class weights corresponding to the
                model labels for the training set in the case of imbalanced
                datasets. Set to `None` to use equal weighting for all classes.
            val_class_weights: See `train_class_weights`.
            test_class_weights: See `train_class_weights`.
        """
        super().__init__()

        # save all named keyword arguments from subclasses as hyperparameters
        self.save_hyperparameters(frame=inspect.currentframe().f_back)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate
        )

        schedulers = [
            LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.hparams.lambdalr_factor ** epoch,
            ),
            {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams.plateau_patience,
                    factor=self.hparams.plateau_factor,
                ),
                "monitor": "val_loss",
            },
        ]

        return [optimizer], schedulers

    def calculate_loss(
        self, outputs, label_batch, weight=None
    ) -> torch.Tensor:

        loss_func = nn.CrossEntropyLoss(weight=weight).to(label_batch.device)

        loss = loss_func(outputs, label_batch)  # pylint: disable=not-callable

        return loss

    def _step(self, image_batch, label_batch, class_weights):

        outputs = self.forward(image_batch)

        loss = self.calculate_loss(
            outputs,
            label_batch,
            weight=torch.as_tensor(
                class_weights, device=label_batch.device, dtype=torch.float32,
            )
            if class_weights is not None
            else None,
        )

        return outputs, loss

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        image_batch, label_batch, metadata = batch

        outputs, loss = self._step(
            image_batch, label_batch, self.hparams.train_class_weights
        )

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        image_batch, label_batch, metadata = batch

        outputs, loss = self._step(
            image_batch, label_batch, self.hparams.val_class_weights
        )

        self.log("val_loss", loss)

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        image_batch, label_batch, metadata = batch

        outputs, loss = self._step(
            image_batch, label_batch, self.hparams.test_class_weights
        )

        self.log("test_loss", loss)

    def compute_features(self, x):
        return self.model(x)

    def forward(self, x):  # pylint: disable=arguments-differ
        features = self.compute_features(x)

        return self.model.classifier(features)


class ResNet18(ClassificationModel):
    """
    Wrapper around PyTorch's implementation of ResNet18.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        learning_rate: float = 0.001,
        lambdalr_factor: float = 0.9,
        plateau_patience: int = 5,
        plateau_factor: float = 0.2,
        train_class_weights=None,
        val_class_weights=None,
        test_class_weights=None,
        pretrained: bool = False,
    ):
        super().__init__()

        model = resnet18(pretrained=self.hparams.pretrained,)

        # disable last fully-connected layer so we can extract features later

        in_features = model.fc.in_features

        model.fc = nn.Identity()

        model.classifier = nn.Linear(
            in_features=in_features, out_features=self.hparams.num_classes,
        )

        self.model = model


class ResNet18Embeddings(ClassificationModel):
    """
    Sequentially operate on input channels independently and merge later.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_channels: int,
        learning_rate: float = 0.001,
        lambdalr_factor: float = 0.9,
        plateau_patience: int = 5,
        plateau_factor: float = 0.2,
        train_class_weights=None,
        val_class_weights=None,
        test_class_weights=None,
        pretrained: bool = False,
    ):
        super().__init__()

        model = resnet18(pretrained=self.hparams.pretrained,)

        # disable last fully-connected layer so we can extract features later

        in_features = model.fc.in_features

        model.fc = nn.Identity()

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(num_channels * in_features, in_features, bias=False,),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=in_features, out_features=self.hparams.num_classes,
            ),
        )

        self.model = model

    def compute_features(self, x):
        all_features = []

        for channel_idx in range(x.shape[1]):
            x_channel = x[:, channel_idx, ...].unsqueeze(1).tile(1, 3, 1, 1)

            features = self.model(x_channel)

            all_features.append(features)

        return torch.cat(all_features, 1)


class ResNet101(ClassificationModel):
    """
    Wrapper around PyTorch's implementation of ResNet101.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        learning_rate: float = 0.001,
        lambdalr_factor: float = 0.9,
        plateau_patience: int = 5,
        plateau_factor: float = 0.2,
        train_class_weights=None,
        val_class_weights=None,
        test_class_weights=None,
        pretrained: bool = False,
    ):
        super().__init__()

        model = resnet101(pretrained=self.hparams.pretrained,)

        # disable last fully-connected layer so we can extract features later

        in_features = model.fc.in_features

        model.fc = nn.Identity()

        model.classifier = nn.Linear(
            in_features=in_features, out_features=self.hparams.num_classes,
        )

        self.model = model
