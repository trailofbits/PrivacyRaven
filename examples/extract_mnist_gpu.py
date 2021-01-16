"""
This model extraction attack steals a model trained on MNIST by
using the copycat synthesizer and the EMNIST dataset to train a
FourLayerClassifier substitute. A single GPU is assumed.
"""
import privacyraven as pr

from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier


# PrivacyRaven's builtin victim functions allow the user to specify the number of GPUs
# to use during computation via the `gpus` keyword argument.  By default, this is 1, but 
# setting gpus=torch.cuda.device_count() will let PrivacyRaven / Pytorch use all available GPUs.
# if torch.cuda.device_count() returns 0, then PrivacyRaven will run in CPU-only mode.

# train_four_layer_mnist_victim() is a PrivacyRaven builtin function that 
# trains a 4-layer fully connected neural network on MNIST data.  See 
# src/privacyraven/models/victims.py for a full set of supported parameters.

model = train_four_layer_mnist_victim(gpus=1)

# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a model extraction attack
attack = ModelExtractionAttack(
    query_mnist,
    200,  # Less than the number of MNIST data points: 60000
    (1, 28, 28, 1),
    10,
    (3, 1, 28, 28),  # Shape of an EMNIST data point
    "copycat",
    FourLayerClassifier,
    784,  # 28 * 28 or the size of a single image
    emnist_train,
    emnist_test,
)
