import torch
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from privacyraven.utils.model_creation import NewDataset, set_evasion_model
from privacyraven.utils.query import reshape_input

synths = dict()


def register_synth(func):
    """Register a function as a synthesizer"""
    synths[func.__name__] = func
    return func


def synthesize(func_name, seed_data_train, seed_data_test, *args, **kwargs):
    """Synthesize training and testing data for a substitute model

    Parameters:
        func_name: String of the function name
        seed_data_train: Tuple of tensors or tensor of training data
        seed_data_test: Tuple of tensors or tensor of training data

    Returns:
        Three NewDatasets containing synthetic data"""
    func = synths[func_name]
    print("Time to synthesize")
    x_train, y_train = func(seed_data_train, *args, **kwargs)
    x_test, y_test = func(seed_data_test, *args, **kwargs)
    print("Synthesis complete")

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.4, random_state=42
    )

    synth_train = NewDataset(x_train, y_train)
    synth_valid = NewDataset(x_valid, y_valid)
    synth_test = NewDataset(x_test, y_test)
    return synth_train, synth_valid, synth_test


def get_data_limit(data):
    """Uses the size of the data to establish a synthesis restriction"""
    try:
        # Differentiate between labeled and unlabeled data
        x_i, y_i = data.data, data.targets
        data_limit = x_i.size()
    except Exception:
        # This requires data to have a size attribute
        data_limit = data.size()
    data_limit = int(data_limit[0])
    return data_limit


@register_synth
def copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    """Creates a synthetic dataset by labeling unlabeled seed data

    Arix Paper: https://ieeexplore.ieee.org/document/8489592"""
    data_limit = get_data_limit(data)

    # The limit must be lower than or equal to the number of queries
    if data_limit > query_limit:
        limit = query_limit
    else:
        limit = data_limit

    # print(limit)

    for i in tqdm(range(0, limit)):
        if i == 0:
            # First assume that the data is in a tuple-like format
            try:
                x, y0 = data[0]
            except Exception:
                x = data[0]
            # Creates new tensors
            y = torch.tensor([query(x)])
            x = reshape_input(x, substitute_input_shape)
        else:
            try:
                xi, y0 = data[i]
            except Exception:
                xi = data[i]
            # Concatenates current data to new tensors
            xi = reshape_input(xi, substitute_input_shape)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))

    # print(f"Dataset Created: {x.shape}; {y.shape}")
    print(x.dtype)
    print(y.dtype)
    return x, y


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
    config = set_evasion_model(query, victim_input_shape, victim_input_targets)
    internal_limit = int(query_limit * 0.5)
    evasion_limit = int(query_limit * 0.5)
    attack = HopSkipJump(
        config,
        False,
        norm="inf",
        max_iter=evasion_limit,
        max_eval=evasion_limit,
        init_eval=10,
    )
    X, y = copycat(
        data,
        query,
        internal_limit,
        victim_input_shape,
        substitute_input_shape,
        victim_input_targets,
    )
    print(X.shape)
    result = attack.generate(X)
    result = torch.as_tensor(result)
    result = result.clone().detach()
    print(result.shape)
    y = torch.Tensor([query(x) for x in result])
    y = y.long()
    return result, y
