import numpy as np
import torch
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pytorch_lightning.metrics.utils import to_onehot

from privacyraven.utils.model_creation import NewDataset, set_evasion_model
from privacyraven.utils.query import reshape_input

# Creates an empty dictionary for synthesis functions
synths = dict()


def register_synth(func):
    """Register a function as a synthesizer"""
    synths[func.__name__] = func
    return func


def synthesize(
    func_name, seed_data_train, seed_data_test, query, query_limit, *args, **kwargs
):
    """Synthesizes training and testing data for a substitute model

    First, the data is processed. Then, the synthesizer function is called.

    Parameters:
        func_name: String of the function name
        seed_data_train: Tuple of tensors or tensor of training data
        seed_data_test: Tuple of tensors or tensor of training data

    Returns:
        Three NewDatasets containing synthetic data"""

    func = synths[func_name]

    # We split the query limit in half to account for two datasets.
    query_limit = int(0.5 * query_limit)

    seed_data_train = process_data(seed_data_train, query_limit)
    seed_data_test = process_data(seed_data_test, query_limit)

    x_train, y_train = func(seed_data_train, query, query_limit, *args, **kwargs)
    x_test, y_test = func(seed_data_test, query, query_limit, *args, **kwargs)
    print("Synthesis complete")

    # Presently, we have hard-coded specific values for the test-train split.
    # In the future, this should be automated and/or optimized in some form.
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.4, random_state=42
    )

    # The NewDataset ensures the synthesized data is a valid PL network input.
    synth_train = NewDataset(x_train, y_train)
    synth_valid = NewDataset(x_valid, y_valid)
    synth_test = NewDataset(x_test, y_test)
    return synth_train, synth_valid, synth_test


def process_data(data, query_limit):
    """Returns x and (if given labeled data) y tensors that are shortened
    to the length of the query_limit if applicable"""

    try:
        # See if the data is labeled regardless of specific representation
        labeled = True
        x, y = data[0]
    except ValueError:
        # A value error is raised if the data is not labeled
        labeled = False
        if isinstance(data, np.ndarray) is True:
            data = torch.from_numpy(data)
        x_data = data.detach().clone().float()
        y_data = None
        bounded = False
    # Labeled data can come in multiple data formats, including, but
    # not limited to Torchvision datasets, lists of tuples, and
    # tuple of tuples. We attempt to address these edge cases
    # through the exception of an AttributeError
    if labeled:
        try:
            if isinstance(data.data, np.ndarray) is True:
                x_data, y_data = (
                    torch.from_numpy(data.data).detach().clone().float(),
                    torch.from_numpy(data.targets).detach().clone().float(),
                )
            else:
                x_data, y_data = (
                    data.data.detach().clone().float(),
                    data.targets.detach().clone().float(),
                )
            bounded = False
        except AttributeError:
            # Setting 'bounded' increases efficiency as data that
            # will be ignored due to the query limit will not be
            # included in the initial x and y data tensors
            bounded = True

            data_limit = int(len(data))
            if query_limit is None:
                data_limit = query_limit
            limit = query_limit if data_limit > query_limit else data_limit

            data = data[:limit]

            x_data = torch.Tensor([x for x, y in data]).float()
            y_data = torch.Tensor([y for x, y in data]).float()

    if bounded is False:
        data_limit = int(x_data.size()[0])
        if query_limit is None:
            # data_limit = query_limit
            query_limit = data_limit

        limit = query_limit if data_limit > query_limit else data_limit

        # torch.narrow is more efficient than indexing and splicing
        x_data = x_data.narrow(0, 0, int(limit))
        if y_data is not None:
            y_data = y_data.narrow(0, 0, int(limit))
            # y_data = to_onehot(y_data)
    # print("Data has been processed")
    processed_data = (x_data, y_data)
    return processed_data


@register_synth
def copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
    reshape=True,
):
    """Creates a synthetic dataset by labeling seed data

    Arxiv Paper: https://ieeexplore.ieee.org/document/8489592"""
    (x_data, y_data) = data
    y_data = query(x_data)
    # y_data = to_onehot(y_data, victim_input_targets)
    if reshape:
        x_data = reshape_input(x_data, substitute_input_shape)
    return x_data, y_data


@register_synth
def hopskipjump(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    """Runs the HopSkipJump evasion attack

    Arxiv Paper: https://arxiv.org/abs/1904.02144"""

    internal_limit = int(query_limit * 0.5)
    X, y = copycat(
        data,
        query,
        internal_limit,
        victim_input_shape,
        substitute_input_shape,
        victim_input_targets,
        False,
    )

    X_np = X.detach().clone().numpy()

    config = set_evasion_model(query, victim_input_shape, victim_input_targets)

    evasion_limit = int(query_limit * 0.5)

    # The initial evaluation number must be lower than the maximum

    lower_bound = 0.01 * evasion_limit

    init_eval = int(lower_bound if lower_bound > 1 else 1)

    attack = HopSkipJump(
        config,
        False,
        norm="inf",
        max_iter=evasion_limit,
        max_eval=evasion_limit,
        init_eval=init_eval,
    )

    print(attack)

    result = torch.from_numpy(attack.generate(X_np)).detach().clone().float()

    y = query(result)

    result = reshape_input(result, substitute_input_shape)
    return result, y
