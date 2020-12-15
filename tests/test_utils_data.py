# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import privacyraven.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset
import torchvision.datasets.mnist
from hypothesis import given, strategies as st

# TODO: replace st.nothing() with appropriate strategies


def valid_hparams():
    hparams = {
        "transform": None,
        "batch_size": 100,
        "num_workers": 4,
        "rand_split_val": [55000, 5000],
        "gpus": 1,
        "max_epochs": 10,
        "learning_rate": 0.001,
        "input_size": 1000,
        "targets": 10,
    }
    return hparams


@given(transform=st.just(None), RGB=st.booleans())
def get_emnist_data_returns_data(transform, RGB):
    emnist_train, emnist_test = privacyraven.utils.data.get_emnist_data(
        transform=transform, RGB=RGB
    )
    x, y = emnist_train.data, emnist_train.targets
    assert x.size() == torch.Size([240000, 28, 28])


@given(hparams=valid_hparams())
def get_mnist_loaders(hparams):
    x, y, z = privacyraven.utils.data.get_mnist_loaders(hparams=hparams)
