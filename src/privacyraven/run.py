from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.extraction.synthesis import synths


def run_all_extraction(
    query,
    victim_input_size,
    substitute_input_size,
    query_limit,
    seed_data_train,
    seed_data_test,
    retrain,
):
    """Run all extraction attacks"""
    for s in synths:
        ModelExtractionAttack(
            query,
            s,
            victim_input_size,
            substitute_input_size,
            query_limit,
            seed_data_train,
            seed_data_test,
            retrain,
        )
