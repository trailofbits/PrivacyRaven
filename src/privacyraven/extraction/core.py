import attr
import torch
from torch.utils.data import DataLoader

from privacyraven.extraction.synthesis import synthesize, synths
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.utils.model_creation import (
    convert_to_inference,
    set_hparams,
    train_and_test,
)
from privacyraven.utils.query import establish_query


@attr.s
class ModelExtractionAttack(object):
    """Defines and launches a model extraction attack

    This is a black-box label-only extraction attack. A keyword representing
    a synthesis functions enables the creation of synthetic data, which is then
    used to create a substitute model. Presently, this class does not perform
    substitute model retraining.

    Attributes:
        query: Function that queries deep learning model
        query_limit: Int of amount of times the model can be queried
        victim_input_shape: Tuple of ints describing the shape of victim inputs
        victim_output_targets: Int of number of labels
        substitute_input_shape: Tuple of ints describing shape of data accepted
        by the substitute model
        substitute_model_arch: PyTorch module of substitute architecture.
                               This can be found in models/pytorch.py
        substitute_input_size: Int of input size for the substitute model
        seed_data_train: Tuple of tensors or tensors of seed data
        seed_data_test: Same as above for test data
        transform: A torchvision.transform to be applied to the data
        batch_size: Int stating how many samples are in a batch of data
        num_workers: Int of the number of workers used in training
        max_epochs: Int of the maximum number of epochs used to train the model
        learning_rate: Float of the learning rate of the model"""

    query = attr.ib()
    query_limit = attr.ib(default=100)
    victim_input_shape = attr.ib(default=None)
    victim_output_targets = attr.ib(default=None)
    substitute_input_shape = attr.ib(default=None)
    synthesizer = attr.ib(default="copycat")
    substitute_model_arch = attr.ib(default=ImagenetTransferLearning)
    substitute_input_size = attr.ib(default=1000)
    seed_data_train = attr.ib(default=None)
    seed_data_test = attr.ib(default=None)

    transform = attr.ib(default=None)
    batch_size = attr.ib(default=100)
    num_workers = attr.ib(default=4)
    gpus = attr.ib(default=1)
    max_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=1e-3)

    synth_train = attr.ib(init=False)
    synth_valid = attr.ib(init=False)
    synth_test = attr.ib(init=False)
    hparams = attr.ib(init=False)
    train_dataloader = attr.ib(init=False)
    valid_dataloader = attr.ib(init=False)
    test_dataloader = attr.ib(init=False)
    substitute_model = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.query = establish_query(self.query, self.victim_input_shape)
        self.synth_train, self.synth_valid, self.synth_test = self.synthesize_data()
        print("Synthetic Data Generated")
        self.hparams = self.set_substitute_hparams()
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self.set_dataloaders()

        self.substitute_model = self.get_substitute_model()

    def synthesize_data(self):
        return synthesize(
            self.synthesizer,
            self.seed_data_train,
            self.seed_data_test,
            self.query,
            self.query_limit,
            self.victim_input_shape,
            self.substitute_input_shape,
            self.victim_output_targets,
        )

    def set_substitute_hparams(self):
        # print("Setting substitute hyperparameters")
        hparams = set_hparams(
            self.transform,
            self.batch_size,
            self.num_workers,
            # None,
            self.gpus,
            self.max_epochs,
            self.learning_rate,
            self.substitute_input_size,
            self.victim_output_targets,
        )
        return hparams

    def set_dataloaders(self):
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
        # print(train_dataloader, valid_dataloader, test_dataloader)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_substitute_model(self):
        model = train_and_test(
            self.substitute_model_arch,
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
            self.hparams,
        )
        model = convert_to_inference(model)
        return model


"""
class ModelExtractionAttack:
    def __init__(
        self,
        query,
        query_limit=100,
        victim_input_shape=None,
        victim_output_targets=None,  # (targets)
        substitute_input_shape=None,
        synthesizer="copycat",
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
        super(ModelExtractionAttack, self).__init__()

        print("Executing model extraction")

        # Extend the query function to incorporate reshaping or resizing the input
        self.query = establish_query(query, victim_input_shape)

        # Directly assign parameters
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
            self.victim_output_targets,
        )

    def set_substitute_hparams(self):
        print("Setting substitute hyperparameters")
        hparams = set_hparams(
            self.transform,
            self.batch_size,
            self.num_workers,
            # None,
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
        print(train_dataloader, valid_dataloader, test_dataloader)
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
"""
