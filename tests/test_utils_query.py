# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.
import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import privacyraven.extraction.synthesis
import privacyraven.utils.query
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.utils import model_creation

# Establish strategies

device = torch.device("cpu")

model = train_four_layer_mnist_victim(gpus=torch.cuda.device_count())


def query_mnist(input_data):
    return privacyraven.utils.query.get_target(model, input_data, (1, 28, 28, 1))


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())


@given(query_func=valid_query(), input_size=st.just((1, 28, 28, 1)))
def test_fuzz_establish_query(query_func, input_size):
    x = privacyraven.utils.query.establish_query(
        query_func=query_func, input_size=input_size
    )

    assert callable(x) is True


@settings(deadline=None)
@given(
    model=st.just(model), input_data=valid_data(), input_size=st.just((1, 28, 28, 1))
)
def test_fuzz_get_target(model, input_data, input_size):
    input_data = torch.from_numpy(input_data)
    target = privacyraven.utils.query.get_target(
        model=model, input_data=input_data, input_size=input_size
    )
    assert torch.argmax(target) >= 0
    assert torch.argmax(target) < 10


@settings(deadline=None)
@given(
    input_data=valid_data(),
    input_size=st.just((1, 28, 28, 1)),
    single=st.just(False),
    warning=st.just(False),
)
def test_fuzz_reshape_input(input_data, input_size, single, warning):
    x = privacyraven.utils.query.reshape_input(
        input_data=input_data, input_size=input_size, single=single, warning=warning
    )
    # assert x.size() == torch.Size([1, 28, 28, 1])
