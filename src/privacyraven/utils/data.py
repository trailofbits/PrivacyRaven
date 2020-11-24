import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import EMNIST, MNIST
from tqdm import tqdm


def is_combined(data):
    """Determines whether or not the dataset has both x and y data"""
    try:
        combined = True
        #X, Y = data.data, data.targets
        x, y = data[0]
    except ValueError:
        combined = False
    return combined

def get_emnist_data(transform=None, RGB=True):
    """Returns EMNIST train and test datasets.

    This function is assumed to be primarily used as seed data.
    DataLoaders and data splits are in synthesis.py

    Parameters:
        transform: Relevant Torchvision transforms to apply to EMNIST
        RGB: A boolean value that decides if the images are RGB

    Returns:
        Two Torchvision datasets with the EMNIST train and test sets
    """

    if transform is None and (RGB is True):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif transform is None and (RGB is False):
        transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    emnist_train = EMNIST(
        os.getcwd(), split="digits", train=True, download=True, transform=transform
    )
    emnist_test = EMNIST(
        os.getcwd(), split="digits", train=False, download=True, transform=transform
    )
    return emnist_train, emnist_test


def get_mnist_loaders(hparams):
    """Returns MNIST DataLoaders from hyperparams in a dict"""
    mnist_train, mnist_val, mnist_test = get_mnist_data(hparams)
    train_dataloader = DataLoader(
        mnist_train,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
    )

    val_dataloader = DataLoader(
        mnist_val, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"]
    )

    test_dataloader = DataLoader(
        mnist_test, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"]
    )
    return train_dataloader, val_dataloader, test_dataloader


def get_mnist_data(hparams):
    """Returns MNIST train and test sets from hyperparams in a dict"""
    mnist_train = MNIST(
        os.getcwd(), train=True, download=True, transform=hparams["transform"]
    )
    mnist_test = MNIST(
        os.getcwd(), train=False, download=True, transform=hparams["transform"]
    )
    mnist_train, mnist_val = random_split(mnist_train, hparams["rand_split_val"])
    return mnist_train, mnist_val, mnist_test
