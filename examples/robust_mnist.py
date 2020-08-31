import numpy as np
import torch

# from torchvision.datasets import MNIST
# from torchvision import models, transforms
import privacyraven as pr
from privacyraven.utils.data import get_emnist_data, get_mnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target, establish_query
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from sklearn import preprocessing

# Create a query function for a PyTorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    input_data = torch.from_numpy(input_data)
    return get_target(model, input_data)


emnist_train, emnist_test = get_emnist_data()

test = BlackBoxClassifier(
    predict=query_mnist,
    input_shape=(1, 28, 28, 1),
    nb_classes=10,
    clip_values=(0, 255),
    preprocessing_defences=None,
    postprocessing_defences=None,
    preprocessing=None,
)

attack = HopSkipJump(test, False, max_iter=50, max_eval=100, init_eval=10)

X, y = emnist_train.data, emnist_train.targets

X = X.to(torch.float32)

X = X.unsqueeze(3)

attack.generate(X)
