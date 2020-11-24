import torch
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from privacyraven.utils.data import is_combined
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
    # import pdb; pdb.set_trace()
    seed_data_train = synthesize(seed_data_train)
    seed_data_test = synthesize(seed_data_test)
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

def process_data(data):
    try:
        # See if the data is labeled regardless of specific representation
        labeled = True
        x, y = data[0]
    except ValueError:
        # A value error is raised if the data is not labeled
        labeled = False
        x_data = data.detach().clone().float()
        y_data = None
        bounded = False
    # Labeled data can come in multiple data formats, including, but
    # not limited to Torchvision datasets, lists of tuples, and
    # tuple of tuples
    if labeled:
        try:
            x_data, y_data = data.data.detach().clone().float(), data.targets.detach().clone().float()
            bounded = False
        except AttributeError:
            bounded = True

            data_limit = int(len(data))
            limit = query_limit if data_limit > query_limit else data_limit
            data = data[:limit]

            x_data = torch.FloatTensor([x for x, y in data])
            y_data = torch.FloatTensor([y for x, y in data])


    if not(bounded):
        data_limit = int(x_data.size()[0])
        limit = query_limit if data_limit > query_limit else data_limit
    processed_data = (x_data, y_data)
    return processed_data


def get_data_limit(data, combined=None):
    """Uses the size of the data to establish a synthesis restriction"""
    """
    try:
        # Differentiate between labeled and unlabeled data
        x_i, y_i = data.data, data.targets
        data_limit = x_i.size()
    except Exception:
        # This requires data to have a size attribute
        # data = torch.tensor(data)
        data_limit = data.size()
    data_limit = int(data_limit[0])
    """
    if combined == None:
        combined = is_combined(data)

    #import pdb; pdb.set_trace()
    if combined:
        try:
            x, y = data.data, data.targets
            data_limit = x.size()
        except Exception:
            print("Fill")
            data_limit = 0
            # Swoop in all of the x data 
    else:
        data_limit = data.size()
    data_limit = int(data_limit[0])
    # print("Data limit is " + str(data_limit))
    return data_limit


@register_synth
def new_copycat(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    (x_data, y_data) = data

    for x in x_data:
        # How do I efficiently do this? map, filter, reduce?
        y = 
        x = reshape_input(x, substitute_input_shape)



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
    # import pdb; pdb.set_trace();
    data_limit = get_data_limit(data)

    # The limit must be lower than or equal to the number of queries
    if data_limit > query_limit:
        limit = query_limit
    else:
        limit = data_limit

    # print(limit)
    # import pdb; pdb.set_trace();
    for i in tqdm(range(0, limit)):
        if i == 0:
            # First assume that the data is in a tuple-like format
            try:
                x, y0 = data[0]
                # print(x.size())
            except Exception:
                # print(data.size())
                # x = data[0]
                x = data[-1:]
                x = x.type(torch.FloatTensor)
                # print(x.size())
            # Creates new tensors
            y = torch.tensor([query(x)])
            x = reshape_input(x, substitute_input_shape)
        else:
            try:
                xi, y0 = data[i]
            except Exception:
                xi = data[-i+1:]
                xi = xi.type(torch.FloatTensor)
            # Concatenates current data to new tensors
            xi = reshape_input(xi, substitute_input_shape)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))

    # print(f"Dataset Created: {x.shape}; {y.shape}")
    # print(x.dtype)
    # print(y.dtype)
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
    # print(X.shape)
    result = attack.generate(X)
    result = torch.as_tensor(result)
    result = result.clone().detach()
    # print(result.shape)
    y = torch.Tensor([query(x) for x in result])
    y = y.long()
    return result, y
