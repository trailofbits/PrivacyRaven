from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from privacyraven.query import reshape_input
from privacyraven.utils import NewDataset

synths = dict()


def register_synth(func):
    """Register a function as a synthesizer"""
    synths[func.__name__] = func
    return func


def synthesize(func_name, seed_data_train, seed_data_test, *args, **kwargs):
    """Synthesize training and testing data for a substitute model"""
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
def knockoff(data, query, query_limit, victim_input_shape, substitute_input_shape):
    """Execute the synthetic data generation phase of the KnockOff Nets attack"""

    for i in tqdm(range(0, query_limit)):
        if i == 0:
            x, y0 = data[0]
            y = torch.tensor([query(x)])
            x = reshape_input(x, substitute_input_shape)
        else:
            xi, y0 = data[i]
            xi = reshape_input(xi, substitute_input_shape)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))

    print(f"Dataset Created: {x.shape}; {y.shape}")
    return x, y


@register_synth
def adv_boost(data, query, query_limit, victim_input_shape, substitute_input_shape):

    x, y = knockoff(
        data, query, query_limit, victim_input_shape, substitute_input_shape
    )
    pass
