import privacyraven as pr
import torch
from privacyraven.extraction.synthesis import register_synth
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.utils.query import reshape_input
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier


# Trains a 4-layer fully connected neural network on MNIST data using all of the GPUs
# available to the user, or CPU if no GPUs are available (torch.cuda.device_count handles this).

model = train_four_layer_mnist_victim(gpus=torch.cuda.device_count())


# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    return get_target(model, input_data, (1, 28, 28, 1))

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()


# Users may define a function to generate synthetic data via a function of the form
#
#   func(seed_data_train, query, query_limit, *args, **kwargs)
#
# This function then can be registered via the @register_synth decorator
# See the following example, which is an aliased version of the copycat synthesizer
# that may be found in privacyraven.extraction.synthesis: 

@register_synth
def custom_synthesizer(data, query, query_limit, victim_input_shape, substitute_input_shape, reshape=True):
    """Creates a synthetic dataset by labeling seed data"""
    (x_data, y_data) = data
    y_data = query(x_data)
    if reshape:
        x_data = reshape_input(x_data, substitute_input_shape)
    return x_data, y_data

# Gets name of synthesizer function.
attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),
    victim_output_targets=10,
    substitute_input_shape=(3, 1, 28, 28),
    synthesizer="custom_synthesizer",
    substitute_model_arch=FourLayerClassifier,
    substitute_input_size=784,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=1,
)



