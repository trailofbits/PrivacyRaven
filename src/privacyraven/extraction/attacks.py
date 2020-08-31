from privacyraven.extraction.core import ModelExtractionAttack


def cloudleak(
    query,
    query_limit,
    victim_input_shape,
    victim_output_targets,
    substitute_input_shape,
    s,
    substitute_model,
    substitute_input_size,
    seed_data_train,
    seed_data_test,
    transform,
    batch_size,
    num_workers,
    gpus,
    max_epochs,
    learning_rate,
):

    """Run CloudLeak model extraction attacks

    CloudLeak was introduced in this paper:---
    This function uses different adversarial attack-based synthesis
    functions to comply with the label-only black-box threat model.These
    attacks aim for a functionally equivalent model and perform
    best on binary classifiers."""

    adv_synths = ["HopSkipJump"]

    cloudleak_attacks = []

    for s in adv_synths:
        attack = ModelExtractionAttack(
            query,
            query_limit,
            victim_input_shape,
            victim_output_targets,
            substitute_input_shape,
            s,
            substitute_model,
            substitute_input_size,
            seed_data_train,
            seed_data_test,
            transform,
            batch_size,
            num_workers,
            gpus,
            max_epochs,
            learning_rate,
        )
        cloudleak_attacks = cloudleak_attacks.append(attack)

    return cloudleak_attacks


def copycats(
    query,
    query_limit,
    victim_input_shape,
    victim_output_targets,
    substitute_input_shape,
    substitute_model,
    substitute_input_size,
    seed_data_train,
    seed_data_test,
    transform,
    batch_size,
    num_workers,
    gpus,
    max_epochs,
    learning_rate,
):
    """Runs a CopyCat attack

    Arxiv Paper: https://arxiv.org/abs/1806.05476

    Presently, this function excludes subset sampling strategies"""

    synthesizer = "copycat"

    attack = ModelExtractionAttack(
        query,
        query_limit,
        victim_input_shape,
        victim_output_targets,
        substitute_input_shape,
        synthesizer,
        substitute_model,
        substitute_input_size,
        seed_data_train,
        seed_data_test,
        transform,
        batch_size,
        num_workers,
        gpus,
        max_epochs,
        learning_rate,
    )

    return attack
