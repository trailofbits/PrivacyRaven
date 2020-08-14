import privacyraven as pr
from privacyraven.data import get_emnist_data
from privacyraven.extraction import ModelExtractionAttack
from privacyraven.query import get_target
from privacyraven.victim import train_mnist_victim
from privacyraven.models import ImagenetTransferLearning

# Create a query function for a PyTorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    return get_target(model, input_data)


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a 'Knockoff Nets' Model Extraction Attack

attack = ModelExtractionAttack(
    query_mnist,
    100,
    (1, 28, 28, 1),
    10,
    (1, 3, 28, 28),
    "knockoff",
    ImagenetTransferLearning,
    1000,
    emnist_train,
    emnist_test,
)
