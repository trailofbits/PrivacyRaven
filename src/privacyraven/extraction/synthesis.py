from art.attacks.evasion import BoundaryAttack, HopSkipJump
from art.estimators.classifiers import BlackBoxClassifier
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


@register_synth
def copycat(data, query, query_limit, victim_input_shape, substitute_input_shape):
    """Generates a dataset from unlabeled data"""

    for i in tqdm(range(0, query_limit)):
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
def hopskipjump(data, query, query_limit, victim_input_shape, substitute_input_shape):
    """Generates a dataset from unlabeled data"""
    copycat_limit = int(query_limit * 0.5)
    x, y = copycat(
        data, query, copycat_limit, victim_input_shape, substitute_input_shape
    )
    x = x.to(torch.float32)
    config = BlackBoxClassifier(
        predict=query,
        input_shape=victim_input_shape,
        nb_classes=10,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
    )
    attack = HopSkipJump(config, False, max_iter=50, max_eval=100, init_eval=10)
    return attack


@register_synth
def seeded_hopskipjump(
    data, query, query_limit, victim_input_shape, substitute_input_shape
):
    pass
    return "null"
