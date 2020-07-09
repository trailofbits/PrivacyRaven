import numpy as np
import pytorch_lightning as pl
import torch


def reshape_query_input(input_data, input_size):
    if input_size is not None:
        try:
            input_data = input_data.reshape(input_size)
        except Exception:
            print("Warning: Data loss is possible during resizing.")
            input_data = input_data.resize_(input_size)
    return input_data


def establish_query(query_func, input_size):
    return lambda input_data: query_func(reshape_query_input(input_data, input_size))


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
    input_data = reshape_query_input(input_data, input_size)
    prediction_as_torch = model(input_data)
    prediction_as_np = prediction_as_torch.cpu().numpy()
    target = int(np.argmax(prediction_as_np))
    return prediction_as_torch, prediction_as_np, target


def get_target(model, input_data, input_size=None):
    """Returns the predicted target of a Pytorch model
    Parameters:
        model (pl.LightningModule): model to be queried
        input_data (Tensor): the x for the model
        input_size (tuple of python:ints): describes shape of x
    Returns:
        target (int): predicted label
    """
    prediction_as_torch, prediction_as_np, target = query_model(
        model, input_data, input_size
    )
    return target
