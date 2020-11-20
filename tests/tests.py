import pytest
import numpy as np
import torch
import privacyraven.extraction.synthesis
from privacyraven.utils import model_creation, query
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.models.victim import train_mnist_victim
from privacyraven.utils.data import get_emnist_data, is_combined
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
    return get_target(model, input_data)


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())

emnist_train, emnist_test = get_emnist_data()

@given(
    data=st.just(emnist_train),    #valid_data(),
    query=valid_query(),
    query_limit=st.just(100),
    victim_input_shape=st.just((1, 28, 28, 1)),
    substitute_input_shape=st.just((1, 3, 28, 28)),
    victim_input_targets=st.just(10),
)
def test_copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    combined = is_combined(data)

    data_limit = privacyraven.extraction.synthesis.get_data_limit(data)

    # The limit must be lower than or equal to the number of queries
    if (data_limit > query_limit) and combined:
        limit = query_limit
        # Slice the x data so we can just iterate over that
        data = torch.tensor((image for i, (x, y) in enumerate(data)))
        print(type(data))
        print(data.size())
    elif (data_limit > query_limit) and not(combined):
        limit = query_limit
    else:
        limit = data_limit

    # import pdb; pdb.set_trace()
    # data = torch.from_numpy(data)
    # print(data.size())
    """
    return privacyraven.extraction.synthesis.copycat(
        data=data,
        query=query,
        query_limit=query_limit,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_input_targets=victim_input_targets,
    )
    """

test_copycat()
