# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import art.attacks.evasion.boundary
import art.attacks.evasion.hop_skip_jump
import privacyraven.extraction.synthesis
import privacyraven.utils.model_creation
import privacyraven.utils.query
from hypothesis import given, strategies as st

# TODO: replace st.nothing() with appropriate strategies


@given(
    estimator=st.nothing(),
    targeted=st.booleans(),
    delta=st.just(0.01),
    epsilon=st.just(0.01),
    step_adapt=st.just(0.667),
    max_iter=st.just(5000),
    num_trial=st.just(25),
    sample_size=st.just(20),
    init_size=st.just(100),
)
def test_fuzz_BoundaryAttack(
    estimator,
    targeted,
    delta,
    epsilon,
    step_adapt,
    max_iter,
    num_trial,
    sample_size,
    init_size,
):
    art.attacks.evasion.boundary.BoundaryAttack(
        estimator=estimator,
        targeted=targeted,
        delta=delta,
        epsilon=epsilon,
        step_adapt=step_adapt,
        max_iter=max_iter,
        num_trial=num_trial,
        sample_size=sample_size,
        init_size=init_size,
    )


@given(
    classifier=st.nothing(),
    targeted=st.booleans(),
    norm=st.just(2),
    max_iter=st.just(50),
    max_eval=st.just(10000),
    init_eval=st.just(100),
    init_size=st.just(100),
)
def test_fuzz_HopSkipJump(
    classifier, targeted, norm, max_iter, max_eval, init_eval, init_size
):
    art.attacks.evasion.hop_skip_jump.HopSkipJump(
        classifier=classifier,
        targeted=targeted,
        norm=norm,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        init_size=init_size,
    )


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


@given(data=st.nothing())
def test_fuzz_get_data_limit(data):
    privacyraven.extraction.synthesis.get_data_limit(data=data)


@given(
    data=st.nothing(),
    query=st.nothing(),
    query_limit=st.nothing(),
    victim_input_shape=st.nothing(),
    substitute_input_shape=st.nothing(),
    victim_input_targets=st.nothing(),
)
def test_fuzz_hopskipjump(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    privacyraven.extraction.synthesis.hopskipjump(
        data=data,
        query=query,
        query_limit=query_limit,
        victim_input_shape=victim_input_shape,
        substitute_input_shape=substitute_input_shape,
        victim_input_targets=victim_input_targets,
    )


@given(func=st.nothing())
def test_fuzz_register_synth(func):
    privacyraven.extraction.synthesis.register_synth(func=func)


@given(
    func_name=st.nothing(), seed_data_train=st.nothing(), seed_data_test=st.nothing()
)
def test_fuzz_synthesize(func_name, seed_data_train, seed_data_test):
    privacyraven.extraction.synthesis.synthesize(
        func_name=func_name,
        seed_data_train=seed_data_train,
        seed_data_test=seed_data_test,
    )


@given(images=st.nothing(), targets=st.nothing(), transform=st.none())
def test_fuzz_NewDataset(images, targets, transform):
    privacyraven.utils.model_creation.NewDataset(
        images=images, targets=targets, transform=transform
    )


@given(
    query=st.nothing(),
    victim_input_shape=st.nothing(),
    victim_input_targets=st.nothing(),
)
def test_fuzz_set_evasion_model(query, victim_input_shape, victim_input_targets):
    privacyraven.utils.model_creation.set_evasion_model(
        query=query,
        victim_input_shape=victim_input_shape,
        victim_input_targets=victim_input_targets,
    )


@given(input_data=st.nothing(), input_size=st.nothing(), warning=st.booleans())
def test_fuzz_reshape_input(input_data, input_size, warning):
    privacyraven.utils.query.reshape_input(
        input_data=input_data, input_size=input_size, warning=warning
    )
