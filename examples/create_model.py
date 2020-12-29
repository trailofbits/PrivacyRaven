import privacyraven as pr
from privacyraven.utils.model_creation import *
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning

# User-defined query function
def query_mnist(input_data):
    return get_target(model, input_data)