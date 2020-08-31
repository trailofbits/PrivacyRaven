from art.attacks.evasion import BoundaryAttack, HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from privacyraven.utils.model_creation import NewDataset
from privacyraven.utils.query import reshape_input

synths = dict()


def register_synth(func):
    """Register a function as a synthesizer"""
    synths[func.__name__] = func
    return func


def synthesize(func_name, seed_data_train, seed_data_test, *args, **kwargs):
    """Synthesize training and testing data for a substitute model"""
    print(synths)
    func = synths[func_name]
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


def set_evasion_model(query, victim_input_shape, victim_input_targets):
    config = BlackBoxClassifier(
        predict=query,
        input_shape=victim_input_shape,
        nb_classes=victim_input_targets,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
    )
    return config


def init_hopskipjump(config, data, limit=50):
    attack = HopSkipJump(config, False, max_iter=limit, max_eval=100, init_eval=10)
    return attack.generate(data)


def get_data_limit(data):
    try:
        x_i, y_i = data.data, data.targets
        data_limit = x_i.size()
    except Exception:
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
    """Generates a dataset from unlabeled data"""
    data_limit = get_data_limit(data)
    if data_limit > query_limit:
        limit = query_limit
    else:
        limit = data_limit
    for i in tqdm(range(0, limit)):
        if i == 0:
            try:
                x, y0 = data[0]
            except Exception:
                x = data[0]
            y = torch.tensor([query(x)])
            x = reshape_input(x, substitute_input_shape)
        else:
            try:
                xi, y0 = data[i]
            except Exception:
                xi = data[i]
            xi = reshape_input(xi, substitute_input_shape)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))

    print(f"Dataset Created: {x.shape}; {y.shape}")
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
    """Generates a dataset from unlabeled data"""
    x, y = copycat(data, query, query_limit, victim_input_shape, substitute_input_shape)
    x = x.to(torch.float32)
    config = set_evasion_model(query, victim_input_shape, victim_input_targets)
    x_adv, y_adv = init_hopskipjump(config, data)
    x = torch.cat((x, x_adv))
    y = torch.cat((y, y_adv))
    return x, y


@register_synth
def seeded_hopskipjump(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    data = data.to(torch.float32)
    config = set_evasion_model(query, victim_input_shape, victim_input_targets)
    x_adv, y_adv = init_hopskipjump(config, data, query_limit)
    return x_adv, y_adv
