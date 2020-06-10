# import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

"""
def train_and_test(
    classifier,
    max_epochs,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    gpus=1,
    num_nodes=1,
):
    # Instantiates, trains, and evaluates the given model on the test set
    # TODO: Take into account every parameter of the PL trainer
    model = classifier()
    trainer = pl.Trainer(gpus=gpus, num_nodes=num_nodes, max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloaders=test_dataloader)
    return model
"""


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
