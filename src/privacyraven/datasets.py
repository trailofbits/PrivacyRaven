import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST


# Define dataset class and loaders
# TODO: Find a better name for this
class CustomDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image.numpy())
        return image, target
