# This test code was modified from code written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import numpy as np
import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import privacyraven.extraction.metrics
import privacyraven.extraction.synthesis
import privacyraven.utils.query
from privacyraven.utils import model_creation
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim

# Establish strategies

device = torch.device("cpu")

model = train_four_layer_mnist_victim(gpus=0)


def query_mnist(input_data):
    return get_target(model, input_data, (1, 28, 28, 1))


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())


@given(
    test_data=valid_data(),
    substitute_model=st.just(model),
    query_victim=valid_query(),
    victim_input_shape=st.just((1, 28, 28, 1)),
    substitute_input_shape=st.just((1, 28, 28, 1)),
)
def label_agreement_returns_agreed(
    test_data,
    substitute_model,
    query_victim,
    victim_input_shape,
    substitute_input_shape,
):
    x = privacyraven.extraction.metrics.label_agreement(
        test_data=test_data,
        substitute_model=substitute_model,
        query_victim=query_victim,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
    )
    # Technically, x should be 10, but that may fail on
    # a less faulty NN- an invariant we should not be
    # testing here

    assert x > 8
