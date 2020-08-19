import numpy as np
import torch
import privacyraven as pr
from privacyraven.data import get_emnist_data
from privacyraven.extraction import ModelExtractionAttack
from privacyraven.query import get_target
from privacyraven.victim import train_mnist_victim
from privacyraven.models import ImagenetTransferLearning
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import BlackBoxClassifier

# Create a query function for a PyTorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    return get_target(model, input_data)


emnist_train, emnist_test = get_emnist_data()

test = BlackBoxClassifier(
    predict=query_mnist,
    input_shape=(1, 28, 28, 1),
    nb_classes=10,
    # clip_values=None,
    # preprocessing_defences=None,
    # postprocessing_defences=None,
    preprocessing=None,
)

attack = BoundaryAttack(test, False)

X, y = emnist_train.data, emnist_train.targets

# X.to(torch.float32)

adv = attack.generate(X, y)

print(adv)
