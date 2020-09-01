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

    Returns an array of attacks that use synthesis functions
    based on adversarial/evasion attacks

    Based upon: https://bit.ly/31Npbgj

    Unlike the paper, this function does not include subset
    sampling strategies and relies upon different evasion
    attacks in order to comply with the threat model"""

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
    """Runs the CopyCat model extraction attack

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
