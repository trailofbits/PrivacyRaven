import privacyraven as pr

import pytorch_lightning as pl
from torch import nn

from privacyraven.utils.model_creation import *
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.pytorch import ImagenetTransferLearning

# User-defined query function
def query_mnist(input_data):
    return get_target(model, input_data)

emnist_train, emnist_test = get_emnist_data()

# Trains a victim model
model = train_four_layer_mnist_victim(gpus=1)

# Gets name of synthesizer function.
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
    gpus=1,
)

