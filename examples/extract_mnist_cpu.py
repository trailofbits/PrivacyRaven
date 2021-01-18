"""
This model extraction attack steals a model trained on MNIST by
using the copycat synthesizer and the EMNIST dataset to train a
FourLayerClassifier substitute. The number of GPUs is set to 0,
and keyword attributes are used for model extraction.
"""
import privacyraven as pr

from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier

# trains a 4-layer fully connected neural network on MNIST data with the user's CPU.  See 
# src/privacyraven/models/victims.py for a full set of supported parameters.

model = train_four_layer_mnist_victim(gpus=0)

# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a model extraction attack
attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),  # EMNIST data point shape
    victim_output_targets=10,
    substitute_input_shape=(3, 1, 28, 28),
    synthesizer="copycat",
    substitute_model_arch=FourLayerClassifier,  # 28*28: image size
    substitute_input_size=784,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=0,
)
