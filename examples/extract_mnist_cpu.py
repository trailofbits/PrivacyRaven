"""
This model extraction attack uses the copycat synthesizer to train
a model pretrained on ImageNet as a pirated MNIST model.
The only requirement of the copycat synthesizer is seed data, which
is achieved by downloading the EMNIST dataset.
This example runs on a CPU, not a GPU.
"""

import privacyraven as pr
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning

# Create a query function for a target PyTorch Lightning model
model = train_mnist_victim(gpus=0)


def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data)


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a Model Extraction Attack
attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),
    victim_output_targets=10,
    substitute_input_shape=(1, 3, 28, 28),
    synthesizer="copycat",
    substitute_model_arch=ImagenetTransferLearning,
    substitute_input_size=1000,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=0,
)
