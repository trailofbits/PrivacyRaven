import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import MNIST

from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.inversion_model import InversionModel
from privacyraven.models.pytorch import ThreeLayerClassifier
from privacyraven.utils.data import get_mnist_loaders
from privacyraven.utils.data import get_prob_loaders
from privacyraven.utils.model_creation import (
    convert_to_inference,
    set_hparams,
    train_and_test,
    train_and_test_inversion,
)

# Trains MNIST inversion model
def train_mnist_inversion(
    classifier,
    transform=None,
    datapoints=None,
    batch_size=100,
    forward_model=None,
    num_workers=4,
    rand_split_val=None,
    gpus=None,
    max_epochs=8,
    inversion_params={"nz": 0, "ngf": 3, "affine_shift": 7, "truncate": 3},
    learning_rate=1e-3,
):
    """Trains a 4-layer fully connected neural network on MNIST data

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

    input_size = 10  # Prediction vector
    targets = 784  # Reconstructed image 

    # Uses all available GPUs for computation by default
    if gpus is None:
        gpus = torch.cuda.device_count()

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
        gpus,
        max_epochs,
        learning_rate,
        input_size,
        targets,
    )

    hparams["rand_split_val"] = rand_split_val
    
    print("Random split: ", rand_split_val)

    train_dataloader, val_dataloader, test_dataloader = get_prob_loaders(hparams, datapoints)
    print("INVERSION:", type(train_dataloader), type(val_dataloader), type(test_dataloader))
    # Train, test, and convert the model to inference
    inversion_model = train_and_test_inversion(
        classifier, InversionModel, train_dataloader, val_dataloader, test_dataloader, hparams, inversion_params
    )
    inversion_model = convert_to_inference(inversion_model)
    return inversion_model



def train_four_layer_mnist_victim(
    transform=None,
    batch_size=100,
    num_workers=4,
    rand_split_val=None,
    gpus=None,
    max_epochs=8,
    learning_rate=1e-3,
):
    """Trains a 4-layer fully connected neural network on MNIST data

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

    input_size = 784  # 28*28 or the size of a single image
    targets = 10  # the number of digits any image can possibly represent

    # Uses all available GPUs for computation by default
    if gpus is None:
        gpus = torch.cuda.device_count()

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
        gpus,
        max_epochs,
        learning_rate,
        input_size,
        targets,
    )

    train_dataloader, val_dataloader, test_dataloader = get_mnist_loaders(hparams)
    # Train, test, and convert the model to inference
    mnist_model = train_and_test(
        FourLayerClassifier, train_dataloader, val_dataloader, test_dataloader, hparams
    )
    mnist_model = convert_to_inference(mnist_model)
    return mnist_model


def train_mnist_victim(
    transform=None,
    batch_size=100,
    num_workers=4,
    rand_split_val=None,
    gpus=None,
    max_epochs=8,
    learning_rate=1e-3,
):
    """Trains a 3-layer fully connected neural network on MNIST data

    This function will be depreciated with the ThreeLayerClassifier.

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

    print(
        "WARNING: The ThreeLayerClassifier will be depreciated. Use the FourLayerClassifier instead."
    )

    # Define hyperparameters implied by the use of MNIST
    input_size = 784
    targets = 10

    # Uses all available GPUs for computation by default
    if gpus is None:
        gpus = torch.cuda.device_count()

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
