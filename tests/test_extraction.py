import pytest

import privacyraven as pr
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning


def test_extraction():
    try:
        print("Creating victim model")
        model = train_mnist_victim()

        def query_mnist(input_data):
            return get_target(model, input_data)

        print("Downloading EMNIST data")
        emnist_train, emnist_test = get_emnist_data()

        print("Launching model extraction attack")
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
            emnist_test,
        )
        print("Attack Done")
        print(attack)
    except Exception:
        pytest.fail("Unexpected Error")
