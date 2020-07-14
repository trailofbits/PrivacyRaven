import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import MNIST

from privacyraven.data import get_mnist_loaders
from privacyraven.models import ThreeLayerClassifier
from privacyraven.utils import convert_to_inference, set_hparams, train_and_test


def train_mnist_victim(
    transform=None,
    batch_size=100,
    num_workers=4,
    rand_split_val=None,
    gpus=1,
    max_epochs=8,
    learning_rate=1e-3,
):
    """Trains a 3-layer fully connected neural network on MNIST data

    Parameters:
        transform: A Torchvision.transforms transformation to be applied to MNIST data
        batch_size: An integer of the size of batches to be trained and tested upon
        num_workers: An integer number of workers assigned to computations
        rand_split_val: An array describing how the val and train data are split
        gpus: An integer num of gpus available to train upon
        max_epochs: An integer of the maximum # of epochs to run
        learning_rate: A float that is the learning rate for the optimizer

    Returns:
        Trained model ready for inference"""
    # Define hyperparameters implied by the use of MNIST

    input_size = 784
    targets = 10
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    if rand_split_val is None:
        rand_split_val = [55000, 5000]

    # Establish hyperparameters and DataLoaders
    hparams = set_hparams(
        transform,
        batch_size,
        num_workers,
        rand_split_val,
        gpus,
        max_epochs,
        learning_rate,
        input_size,
        targets,
    )

    train_dataloader, val_dataloader, test_dataloader = get_mnist_loaders(hparams)
    # Train, test, and convert the model to inference
    mnist_model = train_and_test(
        ThreeLayerClassifier, train_dataloader, val_dataloader, test_dataloader, hparams
    )
    mnist_model = convert_to_inference(mnist_model)
    return mnist_model
