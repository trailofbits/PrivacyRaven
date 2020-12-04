import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import pytest
import numpy as np
import torch
import privacyraven.extraction.synthesis
from privacyraven.utils import model_creation, query
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.models.victim import train_mnist_victim
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import get_target

from hypothesis import assume, given, strategies as st
from hypothesis.extra.numpy import arrays

"""
The synthesis tests rely on sampling data from a model.
We will be training one and returning a query function here
and not inside of a separate function in order to minimize
the cycles dedicated for training this model.
"""

model = train_mnist_victim(gpus=0)


def query_mnist(input_data):
    return get_target(model, input_data, (1, 28, 28, 1))


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())


# emnist_train, emnist_test = get_emnist_data()
# CUDA_LAUNCH_BLOCKING=1


@given(
    data=valid_data(),
    query=st.just(query_mnist),
    query_limit=st.integers(10, 25),
    victim_input_shape=st.just((1, 28, 28, 1)),
    substitute_input_shape=st.just((1, 3, 28, 28)),
    victim_input_targets=st.just(10),
)
def test_copycat_preserves_shapes(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    # import pdb; pdb.set_trace()
    data = torch.from_numpy(data).detach().clone().float()
    data = privacyraven.extraction.synthesis.process_data(data, query_limit)
    x_data, y_data = privacyraven.extraction.synthesis.copycat(
        data=data,
        query=query,
        query_limit=query_limit,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_input_targets=victim_input_targets,
    )
    x_1 = x_data.size()
    y_1 = y_data.size()
    # print(x_1)
    assert x_1 == torch.Size([10, 3, 28, 28])

test_copycat_preserves_shapes()


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
    # device = torch.device("cpu")
    # import pdb; pdb.set_trace()
    # print(data.size())
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


# test_fuzz_hopskipjump()


@given(data=st.nothing(), query_limit=st.nothing())
def test_fuzz_process_data(data, query_limit):
    privacyraven.extraction.synthesis.process_data(data=data, query_limit=query_limit)


@given(func=st.nothing())
def test_fuzz_register_synth(func):
    privacyraven.extraction.synthesis.register_synth(func=func)


@given(
    func_name=st.nothing(),
    seed_data_train=st.nothing(),
    seed_data_test=st.nothing(),
    query=st.nothing(),
    query_limit=st.nothing(),
)
def test_fuzz_synthesize(
    func_name, seed_data_train, seed_data_test, query, query_limit
):
    privacyraven.extraction.synthesis.synthesize(
        func_name=func_name,
        seed_data_train=seed_data_train,
        seed_data_test=seed_data_test,
        query=query,
        query_limit=query_limit,
    )
