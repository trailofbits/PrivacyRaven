# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import privacyraven.utils.query
from hypothesis import given, strategies as st

# TODO: replace st.nothing() with appropriate strategies


def valid_data():
    return arrays(np.float64, (st.integers(), st.integers(), st.integers(), st.integers()), st.floats())

def valid_sizes():
    return tuples(integers(), integers(), integers(), integers())

@given(query_func=st.nothing(), input_size=st.nothing())
def test_fuzz_establish_query(query_func, input_size):
    privacyraven.utils.query.establish_query(
        query_func=query_func, input_size=input_size
    )


@given(model=st.nothing(), input_data=st.nothing(), input_size=st.none())
def test_fuzz_get_target(model, input_data, input_size):
    privacyraven.utils.query.get_target(
        model=model, input_data=input_data, input_size=input_size
    )


@given(model=st.nothing(), input_data=st.nothing(), input_size=st.none())
def test_fuzz_query_model(model, input_data, input_size):
    privacyraven.utils.query.query_model(
        model=model, input_data=input_data, input_size=input_size
    )


@given(
    input_data=valid_data(),
    input_size=st.tuples(),
    single=st.booleans(),
    warning=st.booleans(),
)
def test_fuzz_reshape_input(input_data, input_size, single, warning):
    privacyraven.utils.query.reshape_input(
        input_data=input_data, input_size=input_size, single=single, warning=warning
    )
