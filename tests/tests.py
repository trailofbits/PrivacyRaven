import pytest
import privacyraven.extraction.synthesis
from privacyraven.utils import model_creation, query
from hypothesis import assume, given, strategies as st

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

@given(
    data=st.nothing(),
    query=st.nothing(),
    query_limit=st.nothing(),
    victim_input_shape=st.nothing(),
    substitute_input_shape=st.nothing(),
    victim_input_targets=st.nothing(),
)
def test_fuzz_copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    privacyraven.extraction.synthesis.copycat(
        data=data,
        query=query,
        query_limit=query_limit,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_input_targets=victim_input_targets,
    )


