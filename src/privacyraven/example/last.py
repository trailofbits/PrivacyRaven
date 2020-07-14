"""
# import os
from torchvision import transforms
import privacyraven as pr
from privacyraven.victim import train_mnist_victim
from privacyraven.query import get_target
from privacyraven.data import get_emnist_data
from privacyraven.extraction import ModelExtractionAttack
from privacyraven.synthesize import knockoff

# from torchvision.datasets import MNIST
# import numpy as np
# import torch
# from torch import nn
# from tqdm import tqdm

# from privacyraven.synthesize import knockoff

# Create victim model
model = train_mnist_victim()

# Create query function for model

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
# x, y = mnist_test[25]

"""
"""
def knockoff(data, query, query_limit, victim_input_size, substitute_input_size):
    for i in tqdm(range(0, query_limit)):
        if i == 0:
            x, y0 = data[0]
            y = torch.tensor([query(x)])
            x = x.reshape(substitute_input_size)
        else:
            xi, y0 = data[i]
            xi = xi.reshape(substitute_input_size)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))
    print("Dataset Created: " + str(x.shape) + str(y.shape))
    return x, y
"""
"""


def query_mnist(input_data):
    return get_target(model, input_data, (1, 28, 28, 1))


emnist_train, emnist_test = get_emnist_data()

# x, y = knockoff(emnist_train, query_mnist, 100, (1, 28, 28, 1), (1, 3, 28, 28))

ModelExtractionAttack(query_mnist, knockoff, public_data=emnist_train)

print("Done")

# print(x.shape)
"""
