from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from art.estimators.classification import BlackBoxClassifier
from torch.utils.data import Dataset


def set_evasion_model(query, victim_input_shape, victim_input_targets):
    """Defines the threat model for an evasion attack"""
    config = BlackBoxClassifier(
        predict=query,
        input_shape=victim_input_shape,
        nb_classes=victim_input_targets,
        clip_values=None,  # (0, 255),
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),  # None,
    )
    return config


class NewDataset(Dataset):
    """Creates a Dataset class for PyTorch"""

    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image.numpy())
        return image, target


def set_hparams(
    transform=None,
    batch_size=100,
    num_workers=4,
    gpus=None,
    max_epochs=8,
    learning_rate=1e-3,
    input_size=None,
    targets=None,
):
    """Creates a dictionary of hyperparameters"""
    # This should be optimized
    rand_split_val = [55000, 5000]

    if gpus is None:
        gpus = torch.cuda.device_count()

    if (input_size is None) or (targets is None):
        return "Input size and number of targets need to be defined"
    hparams = {}
    hparams["transform"] = transform
    hparams["batch_size"] = int(batch_size)
    hparams["num_workers"] = int(num_workers)
    hparams["rand_split_val"] = rand_split_val
    hparams["gpus"] = int(gpus)
    hparams["max_epochs"] = int(max_epochs)
    hparams["learning_rate"] = learning_rate
    hparams["input_size"] = input_size
    hparams["targets"] = targets
    return hparams


def train_and_test(
    classifier,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    hparams,
    callback=None,
    trainer_args=None,
):
    """Trains, validates, and tests a PyTorch Lightning model"""
    model = classifier(hparams)
    # We need to configure how the Trainer interprets callbacks and extra arguments.
    # This should be refactored to be more dynamic and priorize the most common cases.
    if callback is not None and trainer_args is not None:
        trainer = pl.Trainer(
            gpus=hparams["gpus"],
            max_epochs=hparams["max_epochs"],
            callbacks=[callback],
            **trainer_args
        )
    elif callback is not None and trainer_args is None:
        trainer = pl.Trainer(
            gpus=hparams["gpus"], max_epochs=hparams["max_epochs"], callbacks=[callback]
        )
    elif callback is None and trainer_args is None:
        trainer = pl.Trainer(gpus=hparams["gpus"], max_epochs=hparams["max_epochs"])
    elif callback is None and trainer_args is not None:
        trainer = pl.Trainer(
            gpus=hparams["gpus"], max_epochs=hparams["max_epochs"], **trainer_args
        )

    # Runs training, validation, and testing
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloaders=test_dataloader)
    return model


def convert_to_inference(model):
    """Allows a model to be used in an inference setting"""
    model.freeze()
    model.eval()
    with suppress(Exception):
        model.cuda()
    return model


def show_test_image(dataset, idx, cmap="gray"):
    """Shows a single datapoint from a test dataset as an image

    Parameters:
        dataset: A Torch dataset or tuple of the dataset with the image
        idx: An integer of the index of the image position
        cmap: An optional string defining the color map for image

    Returns:
        data sampled and displayed"""
    x, y = dataset[idx]
    plt.imshow(x.numpy()[0], cmap=cmap)
    return x
