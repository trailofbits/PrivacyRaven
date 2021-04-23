"""
This model extraction attack steals a model trained on MNIST by
using the copycat synthesizer and the EMNIST dataset to train a
FourLayerClassifier substitute. A single GPU is assumed.
"""
import privacyraven as pr

from privacyraven.utils.data import get_emnist_data, get_mnist_data
#from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.membership_inf.core import TransferMembershipInferenceAttack
from privacyraven.utils.query import get_target, query_model
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier

# Trains a 4-layer fully connected neural network on MNIST data using the user's GPUs.  See 
# src/privacyraven/models/victims.py for a full set of supported parameters.

model = train_four_layer_mnist_victim(gpus=1)

# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

mnist_train, mnist_val, mnist_test = get_mnist_data()

# a, b = query_model(model, mnist_train[0][0], (1, 28, 28, 1))

# print(a)
# print("Then")
# print(b)

# Run a model extraction attack
attack = TransferMembershipInferenceAttack(
    mnist_train[0][0],
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

# print(attack.__dict__)
