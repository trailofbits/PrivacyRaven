import torch
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from privacyraven.extraction.synthesis import hopskipjump
from privacyraven.utils.model_creation import NewDataset
from privacyraven.utils.query import reshape_input

robust = dict()


def register_robust(func):
    """Register a function as a robustness metric"""
    robust[func.__name__] = func
    return func


def find_robustness(
    func_name,
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
    *args,
    **kwargs
):
    """Synthesize training and testing data for a substitute model"""
    func = robust[func_name]
    X, y = func(*args, **kwargs)
    return X, y


@register_robust
def membership_inf_hopskipjump(
    data,
    query,
    query_limit,
    victim_input_shape,
    substitute_input_shape,
    victim_input_targets,
):
    return hopskipjump(
        data,
        query,
        query_limit,
        victim_input_shape,
        substitute_input_shape,
        victim_input_targets,
    )
