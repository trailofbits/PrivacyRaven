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
    data=st.just(emnist_train),  # valid_data(),
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
    try:
        # See if the data is labeled regardless of specific representation
        labeled = True
        x, y = data[0]
    except ValueError:
        # A value error is raised if the data is not labeled
        labeled = False
        x_data = data.detach().clone().float()
        y_data = None
        bounded = False
    # Labeled data can come in multiple data formats, including, but
    # not limited to Torchvision datasets, lists of tuples, and
    # tuple of tuples
    if labeled:
        try:
            x_data, y_data = data.data.detach().clone().float(), data.targets.detach().clone().float()
            bounded = False
        except AttributeError:
            bounded = True

            data_limit = int(len(data))
            limit = query_limit if data_limit > query_limit else data_limit
            data = data[:limit]

            x_data = torch.FloatTensor([x for x, y in data])
            y_data = torch.FloatTensor([y for x, y in data])


    if not(bounded):
        data_limit = int(x_data.size()[0])
        limit = query_limit if data_limit > query_limit else data_limit


            # return x, y


test_copycat()
