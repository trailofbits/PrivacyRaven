import torch


class ModelExtractionAttack:
    def __init__(
        self,
        query,
        synthesize,
        victim_input_size=(1, 1, 28, 28),
        substitute_input_size=(1, 3, 28, 28),
        query_limit=100,
        public_data=None,
        adv_attack=None,
        retrain=None,
    ):

        super(ModelExtractionAttack, self).__init__()

        self.query = query
        self.synthesize = synthesize
        self.victim_input_size = victim_input_size
        self.query_limit = query_limit
        self.public_data = public_data
        self.adv_attack = adv_attack

        self.x_train, self.y_train = self.synthesize_data()

    def synthesize_data(self):
        return self.synthesize(
            self.query, self.query_limit, self.public_data, self.adv_attack
        )
