"""
This model extraction attack uses the copycat synthesizer to train
a model pretrained on ImageNet as a pirated MNIST model.
The only requirement of the copycat synthesizer is seed data, which
is achieved by downloading the EMNIST dataset.
A single GPU is assumed.
"""

import privacyraven as pr

from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning

# Create a query function for a target PyTorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a model extraction attack
attack = ModelExtractionAttack(
    query_mnist,
    100,
    (1, 28, 28, 1),
    10,
    (1, 3, 28, 28),
    "copycat",
    ImagenetTransferLearning,
    1000,
    emnist_train,
    emnist_test
)

# Use emnist_train.data and emnist_test.data for unlabeled data
