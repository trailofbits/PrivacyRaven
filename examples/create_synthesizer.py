import privacyraven as pr
from privacyraven.extraction.synthesis import (synthesize, register_synth)
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning


# Users may define a function to generate synthetic data via a function of the form
#
#   func(seed_data_train, query, query_limit, *args, **kwargs)
#
# This function then can be registered via the @register_synth decorator
# See the following example (may also be found in synthesis.py): 

@register_synth
def copycat(data, query, query_limit, substitute_input_shape, reshape=True):
    """Creates a synthetic dataset by labeling seed data

    Arxiv Paper: https://ieeexplore.ieee.org/document/8489592"""
    (x_data, y_data) = data
    print("y_data: ", y_data)
    y_data = query(x_data)
    if reshape:
        x_data = reshape_input(x_data, substitute_input_shape)
    return x_data, y_data

# User-defined query function
def query_mnist(input_data):
    print("Input data: ", input_data)
    return get_target(model, input_data)

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Create a query function for a target PyTorch Lightning model
model = train_mnist_victim(gpus=1)

# Gets name of synthesizer function.
func_name = copycat.__name__
substitute_input_shape = (1, 28, 28, 1)

query_limit = 1

synthesize(func_name, emnist_train, emnist_test, query_mnist, query_limit, substitute_input_shape)


