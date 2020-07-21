import torch

from privacyraven.query import establish_query
from privacyraven.synthesis import synthesize, synths


class ModelExtractionAttack:
    def __init__(
        self,
        query,
        synthesizer,
        victim_input_size=(1, 28, 28, 1),
        substitute_input_size=(1, 3, 28, 28),
        query_limit=100,
        seed_data=None,
        retrain=None,
    ):
        """Defines and launches a model extraction attack

        Parameters:
            query: A function that queries the victim model
            synthesizer: A string with the synthesizer name;
                         these names are contained within
                         synths and synthesis.py
            victim_input_size: A tuple describing the size
                               of the victim input data
            substitute_input_size: A tuple with the size of
                                   the substitute inputs
            query_limit: An integer limiting the queries to
                         the victim model
            seed_data: A Torchvision/tuple-like dataset
                         for the Knockoff attack
            retrain: A Boolean value that determines if
                    retraining occurs"""

        super(ModelExtractionAttack, self).__init__()

        self.query = establish_query(query, victim_input_size)
        self.synthesizer = synthesizer
        self.victim_input_size = victim_input_size
        self.substitute_input_size = substitute_input_size
        self.query_limit = query_limit
        self.seed_data = seed_data
        # TODO: Use retrain value

        self.x_train, self.y_train = self.synthesize_data()

    def synthesize_data(self):
        return synthesize(
            self.synthesizer,
            self.seed_data,
            self.query,
            self.query_limit,
            self.victim_input_size,
            self.substitute_input_size,
        )


def run_all_extraction(
    query, victim_input_size, substitute_input_size, query_limit, seed_data, retrain
):
    for s in synths:
        ModelExtractionAttack(
            query,
            s,
            victim_input_size,
            substitute_input_size,
            query_limit,
            seed_data,
            retrain,
        )
