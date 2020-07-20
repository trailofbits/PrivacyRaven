import torch
from tqdm import tqdm

synths = dict()


def register_synth(func):
    """Register a function as a synthesizer"""
    synths[func.__name__] = func
    return func


def synthesize(func_name, *args, **kwargs):
    """Synthesize training and testing data for a substitute model

    TODO: Add dataset and DataLoader creation
    TODO: Handle argument differentiation between synthesizers"""
    func = synths[func_name]
    return func(*args, **kwargs)


@register_synth
def knockoff(data, query, query_limit, victim_input_size, substitute_input_size):
    """Execute the synthetic data generation phase of the KnockOff Nets attack

    TODO: Test efficiency of synthesizers as generators
    TODO: Improve the efficiency of this with vectorization or map/filter/reduce"""
    for i in tqdm(range(0, query_limit)):
        if i == 0:
            x, y0 = data[0]
            y = torch.tensor([query(x)])
            x = x.reshape(substitute_input_size)
        else:
            xi, y0 = data[i]
            xi = xi.reshape(substitute_input_size)
            x = torch.cat((x, xi))
            yi = torch.tensor([query(xi)])
            y = torch.cat((y, yi))
    # print("Dataset Created: " + str(x.shape) + str(y.shape))

    print(f"Dataset Created: {x.shape}; {y.shape}")
    return x, y


# TODO: Add more synthesizers
