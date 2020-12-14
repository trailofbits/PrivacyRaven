# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import privacyraven.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset
import torchvision.datasets.mnist
from hypothesis import given, strategies as st

# TODO: replace st.nothing() with appropriate strategies


@given(transform=st.none(), RGB=st.booleans())
def test_fuzz_get_emnist_data(transform, RGB):
    privacyraven.utils.data.get_emnist_data(transform=transform, RGB=RGB)


@given(hparams=st.nothing())
def test_fuzz_get_mnist_data(hparams):
    privacyraven.utils.data.get_mnist_data(hparams=hparams)


@given(hparams=st.nothing())
def test_fuzz_get_mnist_loaders(hparams):
    privacyraven.utils.data.get_mnist_loaders(hparams=hparams)
