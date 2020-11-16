import pytest

import privacyraven as pr
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.models.victim import train_mnist_victim
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import get_target


def test_extraction():
    try:
        print("Creating victim model")
        model = train_mnist_victim(gpus=0)

        def query_mnist(input_data):
            return get_target(model, input_data)

        print("Downloading EMNIST data")
        emnist_train, emnist_test = get_emnist_data()

        print("Launching model extraction attack")
 
        #This is a CPU-only attack
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
        print(attack)
    except Exception:
        pytest.fail("Unexpected Error")
