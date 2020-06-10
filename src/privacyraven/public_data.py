"""
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST, EMNIST
from torchvision import datasets, transforms, models


def get_emnist_data(transform=None, RGB=True):
    if transform is None and (RGB == True):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif transform is None and (RGB == False):
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
"""
