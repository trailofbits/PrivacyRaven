# This test code was modified from code written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import numpy as np
import pytest
import torch
from art.estimators.classification import BlackBoxClassifier
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import privacyraven.extraction.synthesis
import privacyraven.utils.query
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.utils import model_creation, query
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import get_target

"""
The synthesis tests rely on sampling data from a model.
We will be training one and returning a query function here
and not inside of a separate function in order to minimize
the cycles dedicated for training this model.
"""

device = torch.device("cpu")

model = train_four_layer_mnist_victim(gpus=torch.cuda.device_count())

art_model = BlackBoxClassifier(
    predict=query,
    input_shape=(1, 28, 28, 1),
    nb_classes=10,
    clip_values=None,  # (0, 255),
    preprocessing_defences=None,
    postprocessing_defences=None,
    preprocessing=(0, 1),  # None,
)


def query_mnist(input_data):
    return privacyraven.utils.query.get_target(model, input_data, (1, 28, 28, 1))


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())


@settings(deadline=None)
@given(
    data=valid_data(),
    query=st.just(query_mnist),
    query_limit=st.integers(10, 25),
    art_model=st.just(art_model),
    victim_input_shape=st.just((1, 28, 28, 1)),
    substitute_input_shape=st.just((3, 1, 28, 28)),
    victim_output_targets=st.just(10),
)
def test_copycat_preserves_shapes(
    data,
    query,
    query_limit,
    art_model,
    victim_input_shape,
    substitute_input_shape,
    victim_output_targets,
):
    # data = torch.from_numpy(data).detach().clone().float()
    data = privacyraven.extraction.synthesis.process_data(data, query_limit)
    x_data, y_data = privacyraven.extraction.synthesis.copycat(
        data=data,
        query=query,
        query_limit=query_limit,
        art_model=art_model,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_output_targets=victim_output_targets,
    )
    x_1 = x_data.size()
    y_1 = y_data.size()
    assert x_1 == torch.Size([10, 1, 28, 28])
    assert y_1 == torch.Size([10])


@given(data=valid_data(), query_limit=st.integers(10, 25))
def process_data_preserves_shape_and_type(data, query_limit):
    processed_data = privacyraven.extraction.synthesis.process_data(
        data=data, query_limit=query_limit
    )
    (x, y) = processed_data
    assert x.size() == torch.Size([10, 28, 28, 1])
    assert x.type() == torch.FloatTensor


"""
This is error-prone, but should be fixed eventually

@given(
    data=valid_data(),
    query=st.just(query_mnist),
    query_limit=st.integers(10, 25),
    victim_input_shape=st.just((1, 28, 28, 1)),
    substitute_input_shape=st.just((1, 3, 28, 28)),
    victim_input_targets=st.just(10),
)
def test_fuzz_hopskipjump(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    data = torch.from_numpy(data).detach().clone().float()
    data = privacyraven.extraction.synthesis.process_data(data, query_limit)
    x_data, y_data = privacyraven.extraction.synthesis.hopskipjump(
        data=data,
        query=query,
        query_limit=query_limit,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_input_targets=victim_input_targets,
    )
    print(x_data.size())
    print(y_data.size())
"""
