import numpy as np
import pytorch_lightning as pl
import torch


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
        input_data = input_data.reshape(input_size)
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
