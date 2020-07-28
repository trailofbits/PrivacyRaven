import torch
from torch.utils.data import DataLoader

from privacyraven.query import establish_query
from privacyraven.synthesis import synthesize, synths
from privacyraven.utils import convert_to_inference, set_hparams, train_and_test


class ModelExtractionAttack:
    def __init__(
        self,
        query,
        query_limit=100,
        victim_input_shape=None,
        victim_output_targets=None,  # (targets)
        substitute_input_shape=None,
        synthesizer="Knockoff",
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
        """Defines and launches a model extraction attack"""

        super(ModelExtractionAttack, self).__init__()

        print("Executing model extraction")

        self.query = establish_query(query, victim_input_shape)

        # Is there a way to avoid 14 lines of direct assignment?
        self.query_limit = query_limit
        self.victim_input_shape = victim_input_shape
        self.victim_output_targets = victim_output_targets
        self.substitute_input_size = substitute_input_size
        self.substitute_input_shape = substitute_input_shape
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
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self.set_dataloaders()
        self.substitute_model = self.get_substitute_model()

    def synthesize_data(self):
        print("Synthesizing Data")
        return synthesize(
            self.synthesizer,
            self.seed_data_train,
            self.seed_data_test,
            self.query,
            self.query_limit,
            self.victim_input_shape,
            self.substitute_input_shape,
        )

    def set_substitute_hparams(self):
        print("Setting substitute hyperparameters")
        hparams = set_hparams(
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
        print(hparams)
        return hparams

    def set_dataloaders(self):
        print("Creating DataLoaders")
        train_dataloader = DataLoader(
            self.synth_train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        valid_dataloader = DataLoader(
            self.synth_valid,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        test_dataloader = DataLoader(
            self.synth_test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        return train_dataloader, valid_dataloader, test_dataloader

    def get_substitute_model(self):
        print("Training and testing substitute model")

        model = train_and_test(
            self.substitute_model,
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
            self.hparams,
        )
        model = convert_to_inference(model)
        return model
