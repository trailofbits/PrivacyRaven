import torch

from privacyraven.query import establish_query
from privacyraven.synthesis import synthesize, synths
from privacyraven.utils import set_hparams


class ModelExtractionAttack:
    def __init__(
        self,
        query,
        query_limit=100,
        victim_input_size=None,
        victim_output_targets=None,  # (targets)
        substitute_input_size=None,
        synthesizer="Knockoff",
        substitute_model=None,
        seed_data_train=None,
        seed_data_test=None,
        transform=None,
        batch_size=100,
        num_workers=4,
        gpus=1,
        max_epochs=10,
        learning_rate=1e-3,
    ):
        """Defines and launches a model extraction attack

        Attributes:
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
            seed_data_train: A Torchvision/tuple-like train
                             dataset for the extraction
            seed_data_test: A Torchvision/tuple-like test
                            dataset for the extraction
            retrain: A Boolean value that determines if
                    retraining occurs"""

        super(ModelExtractionAttack, self).__init__()

        self.query = establish_query(query, victim_input_size)

        self.query_limit = query_limit
        self.victim_input_size = victim_input_size
        self.victim_output_targets = victim_output_targets
        self.substitute_input_size = substitute_input_size
        self.synthesizer = synthesizer
        self.substitute_model = substitute_model
        self.seed_data_train = seed_data_train
        self.seed_data_test = seed_data_test
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        self.synth_train, self.synth_valid, self.synth_test = self.synthesize_data()
        self.hparams = self.set_substitute_hparams()

    def synthesize_data(self):
        return synthesize(
            self.synthesizer,
            self.seed_data_train,
            self.seed_data_test,
            self.query,
            self.query_limit,
            self.victim_input_size,
            self.substitute_input_size,
        )

    def set_substitute_hparams(self):
        self.hparams = set_hparams(
            self.transform,
            self.batch_size,
            self.num_workers,
            None,
            self.gpus,
            self.max_epochs,
            self.learning_rate,
            self.substitute_input_size,
            self.victim_output_targets,
        )
        return self.hparams
