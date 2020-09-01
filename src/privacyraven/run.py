from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.extraction.synthesis import synths


def run_all_extraction(
    query,
    query_limit=100,
    victim_input_shape=None,
    victim_output_targets=None,  # (targets)
    substitute_input_shape=None,
    substitute_model=None,
    substitute_input_size=1000,
    seed_data_train=None,
    seed_data_test=None,
    transform=None,
    batch_size=100,
    num_workers=4,
    gpus=1,
    max_epochs=10,
    learning_rate=1e-3,
):
    """Run all extraction attacks"""

    for s in synths:
        ModelExtractionAttack(
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
