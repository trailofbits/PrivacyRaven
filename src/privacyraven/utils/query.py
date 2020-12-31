from contextlib import suppress

import numpy as np
import pytorch_lightning as pl
import torch


def reshape_input(input_data, input_size, single=True, warning=False):
    """Reshape input data before querying model

    This function will conduct a low-level resize if the size of
    the input data is not compatabile with the model input size.

    Parameters:
        input_data: A Torch tensor or Numpy array of the data
        input_size: A tuple of integers describing the new size
        warning: A Boolean that turns warnings on or off

    Returns:
        Data of new shape"""
    with suppress(Exception):
        input_data = torch.from_numpy(input_data)

    if input_size is None:
        if warning is True:
            print("No size was given and no reshaping can occur")
        return input_data

    start = len(input_data)

    alternate = list(input_size)
    alternate[0] = start
    alternate = tuple(alternate)

    try:
        if single:
            input_data = input_data.reshape(alternate)
        else:
            input_data = input_data.reshape(input_size)
    except Exception:
        if warning is True:
            print("Warning: Data loss is possible during resizing.")
        if single:
            input_data = input_data.resize_(alternate)
        else:
            input_data = input_data.resize_(input_size)
    return input_data


def establish_query(query_func, input_size):
    """Equips a query function with the capacity to reshape data"""
    return lambda input_data: query_func(reshape_input(input_data, input_size))


def query_model(model, input_data, input_size=None):
    """Returns the predictions of a Pytorch model

    Parameters:
        model: A pl.LightningModule or Torch module to be queried
        input_data: A Torch tensor entering the model
        input_size: A tuple of ints describes the shape of x

    Returns:
        prediction_as_torch: A Torch tensor of the predicton probabilities
        prediction_as_np: A Numpy array of the predicton probabilities
        target: An integer displaying the predicted label
    """
    # with suppress(Exception):
    # input_data = torch.from_numpy(input_data)
    with suppress(Exception):
        input_data = input_data.cuda()
    input_data = input_data.float()
    if input_size is not None:
        input_data = reshape_input(input_data, input_size)
        # print(input_size)

    # print(input_data.size())

    prediction = model(input_data)
    # print(prediction.size())
    # print(target)
    if prediction.size()[0] == 1:
        print("Single")
        target = torch.argmax(prediction, dim=0, keepdim=True)
    else:
        target = torch.tensor(
            [torch.argmax(row, dim=0, keepdim=True) for row in torch.unbind(prediction)]
        )

    # target = torch.tensor([int(torch.argmax(prediction, dim=0, keepdim=True))])

    return prediction, target


def get_target(model, input_data, input_size=None):
    """Returns the predicted target of a Pytorch model

    Parameters:
        model: A pl.LightningModule or Torch module to be queried
        input_data: A Torch tensor entering the model
        input_size: A tuple of ints describes the shape of x

    Returns:
        target: An integer displaying the predicted label
    """
    prediction, target = query_model(model, input_data, input_size)
    return target
