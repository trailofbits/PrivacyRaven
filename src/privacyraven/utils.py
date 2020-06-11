import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


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


def set_hparams(
    transform,
    batch_size,
    num_workers,
    rand_split_val,
    gpus,
    max_epochs,
    learning_rate,
    input_size,
    targets,
):
    """Creates a dictionary of hyperparameters"""
    hparams = {}
    hparams["transform"] = transform
    hparams["batch_size"] = int(batch_size)
    hparams["num_workers"] = int(num_workers)
    hparams["rand_split_val"] = rand_split_val
    hparams["gpus"] = int(gpus)
    hparams["max_epochs"] = int(max_epochs)
    hparams["learning_rate"] = learning_rate
    hparams["input_size"] = input_size
    hparams["targets"] = targets
    return hparams


def train_and_test(
    classifier, train_dataloader, val_dataloader, test_dataloader, hparams
):
    model = classifier(hparams)
    trainer = pl.Trainer(gpus=hparams["gpus"], max_epochs=hparams["max_epochs"])
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloaders=test_dataloader)
    return model


def convert_to_inference(model):
    """Allows a model to be used in an inference setting"""
    model.freeze()
    model.eval()
    model.cuda()
    return model


def query_model(model, input_data, input_size=None):
    """Returns the predictions of a Pytorch model

    Parameters:
        model (pl.LightningModule): model to be queried
        input_data (Tensor): the x for the model
        input_size (tuple of python:ints): describes shape of x

    Returns:
        prediction (Tensor): predicton probabilities
        np_prediction (Numpy array): predicton probabilities
        target (int): predicted label
    """
    input_data = input_data.cuda()
    if input_size is not None:
        input_data = input_data.reshape(input_data, input_size)
    prediction = model(input_data)
    np_prediction = prediction.cpu().numpy()
    target = int(np.argmax(np_prediction))
    return prediction, np_prediction, target


def get_target(model, input_data, input_size=None):
    """Returns the predicted target of a Pytorch model

    Parameters:
        model (pl.LightningModule): model to be queried
        input_data (Tensor): the x for the model
        input_size (tuple of python:ints): describes shape of x

    Returns:
        target (int): predicted label
    """
    prediction, np_prediction, target = query_model(model, input_data, input_size)
    return target


def show_test_image(dataset, idx, cmap="gray"):
    """Shows a single datapoint from the test set as an image

    Parameters:
        dataset (Torch dataset/Tuple): test dataset to obtain image from
        idx (int): index describing position of image
        cmap (String): optional; defines color map for image

    Returns:
        data sampled and displayed
    """
    x, y = dataset[idx]
    plt.imshow(x.numpy()[0], cmap=cmap)
    return x
