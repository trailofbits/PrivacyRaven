import numpy as np
import pytorch_lightning as pl
import torch


def reshape_input(input_data, input_size, warning=False):
    """Reshape input data before querying model

    This function will conduct a low-level resize if the size of
    the input data is not compatabile with the model input size.

    Parameters:
        input_data: A Torch tensor or Numpy array of the data
        input_size: A tuple of integers describing the new size
        warning: A Boolean that turns warnings on or off

    Returns:
        Data of new shape
    """
    try:
        input_data = torch.from_numpy(input_data)
    except Exception:
        print(input_data.size())
    if input_size is not None:
        try:
            input_data = input_data.reshape(input_size)
        except Exception:
            if warning is True:
                print("Warning: Data loss is possible during resizing.")
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
    try:
        input_data = input_data.cuda()
    except Exception:
        print("Cuda method is not being used")
    input_data = reshape_input(input_data, input_size)
    prediction_as_torch = model(input_data)
    prediction_as_np = prediction_as_torch.cpu().numpy()
    target = int(np.argmax(prediction_as_np))
    return prediction_as_torch, prediction_as_np, target


def get_target(model, input_data, input_size=None):
    """Returns the predicted target of a Pytorch model

    Parameters:
        model: A pl.LightningModule or Torch module to be queried
        input_data: A Torch tensor entering the model
        input_size: A tuple of ints describes the shape of x

    Returns:
        target: An integer displaying the predicted label
    """
    prediction_as_torch, prediction_as_np, target = query_model(
        model, input_data, input_size
    )
    return target
