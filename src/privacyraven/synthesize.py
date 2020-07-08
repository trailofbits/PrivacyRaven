import torch
from tqdm import tqdm


def knockoff(data, query, query_limit, victim_input_size, substitute_input_size):
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
    print("Dataset Created: " + str(x.shape) + str(y.shape))
    return x, y
