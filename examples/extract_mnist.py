import privacyraven as pr
from privacyraven.data import get_emnist_data
from privacyraven.extraction import ModelExtractionAttack
from privacyraven.query import get_target
from privacyraven.victim import train_mnist_victim

model = train_mnist_victim()


def query_mnist(input_data):
    return get_target(model, input_data)


emnist_train, emnist_test = get_emnist_data()

test = ModelExtractionAttack(
    query_mnist, "knockoff", (1, 28, 28, 1), (1, 3, 28, 28), 100, emnist_train
)
