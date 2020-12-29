import privacyraven as pr
from privacyraven.extraction.synthesis import (synthesize, register_synth)
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.utils.query import reshape_input

# Users may define a function to generate synthetic data via a function of the form
#
#   func(seed_data_train, query, query_limit, *args, **kwargs)
#
# This function then can be registered via the @register_synth decorator
# See the following example (may also be found in synthesis.py): 

@register_synth
def custom_synthesizer(data, query, query_limit, victim_input_shape, substitute_input_shape, reshape=True):
    """Creates a synthetic dataset by labeling seed data

    Arxiv Paper: https://ieeexplore.ieee.org/document/8489592"""
    (x_data, y_data) = data
    #print("y_data: ", y_data)
    y_data = query(x_data)
    if reshape:
        x_data = reshape_input(x_data, substitute_input_shape)
    return x_data, y_data

# User-defined query function
def query_mnist(input_data):
    return get_target(model, input_data)

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Create a query function for a target PyTorch Lightning model
model = train_mnist_victim(gpus=1)

# Gets name of synthesizer function.
attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),
    victim_output_targets=10,
    substitute_input_shape=(1, 3, 28, 28),
    synthesizer="custom_synthesizer",
    substitute_model_arch=ImagenetTransferLearning,
    substitute_input_size=1000,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=1,
)



