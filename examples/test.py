"""
This model extraction attack uses the copycat synthesizer to train
a model pretrained on ImageNet as a pirated MNIST model.
The only requirement of the copycat synthesizer is seed data, which
is achieved by downloading the EMNIST dataset.
A single GPU is assumed.
"""
from tqdm import tqdm

import privacyraven as pr
import torch

from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target, reshape_input
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning
from privacyraven.extraction.synthesis import register_synth, get_data_limit, synths

# Create a query function for a target PyTorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data)


@register_synth
def new_copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    """Creates a synthetic dataset by labeling unlabeled seed data
    Arix Paper: https://ieeexplore.ieee.org/document/8489592"""
    print("Hello World")
    # import pdb; pdb.set_trace()
    data_limit = get_data_limit(data)
    #import pdb; pdb.set_trace()

    # The limit must be lower than or equal to the number of queries
    if data_limit > query_limit:
        limit = query_limit
    else:
        limit = data_limit

    # print(limit)

    for i in tqdm(range(0, limit)):
        if i == 0:
            # First assume that the data is in a tuple-like format
            try:
                x, y0 = data[0]
            except Exception:
                x = data[0]
                x = x.type(torch.FloatTensor)
            # Creates new tensors
            y = torch.tensor([query(x)])
            x = reshape_input(x, substitute_input_shape)
        else:
            try:
                xi, y0 = data[i]
            except Exception:
                xi = data[i]
                x = x.type(torch.FloatTensor)
            # Concatenates current data to new tensors
            xi = reshape_input(xi, substitute_input_shape)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))
    return x, y

#print(synths)

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a Model Extraction Attack
attack = ModelExtractionAttack(
    query_mnist,
    100,
    (1, 28, 28, 1),
    10,
    (1, 3, 28, 28),
    "new_copycat",
    ImagenetTransferLearning,
    1000,
    emnist_train.data,
    emnist_test.targets,
)

