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
        public_data=None,
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
            public_data:---
            retrain:---
        """

        super(ModelExtractionAttack, self).__init__()

        self.query = establish_query(query, victim_input_size)
        self.synthesizer = synthesizer
        self.victim_input_size = victim_input_size
        self.substitute_input_size = substitute_input_size
        self.query_limit = query_limit
        self.public_data = public_data
        # self.adv_attack = adv_attack

        self.x_train, self.y_train = self.synthesize_data()

    def synthesize_data(self):
        # Need to add argument differentiation
        return synthesize(self.synthesizer)


"""
        return self.synthesize(
            self.public_data,
            self.query,
            self.query_limit,
            self.victim_input_size,
            self.substitute_input_size,
        )
"""
