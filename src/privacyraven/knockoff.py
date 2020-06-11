import torch
from tqdm import tqdm


# Need to refactor this to facilitate different types of attacks
def knockoff(data, query, model, points, victim_input_size, substitute_input_size):
    for i in tqdm(range(0, points)):
        if i == 0:
            x, y0 = data[0]
            y = torch.tensor([query(model, x, victim_input_size)])
            x = x.reshape(substitute_input_size)
        else:
            xi, y0 = data[i]
            yi = torch.tensor([query(model, xi, victim_input_size)])
            xi = xi.reshape(substitute_input_size)
            x = torch.cat((x, xi))
            y = torch.cat((y, yi))
    print("Dataset Created: " + str(x.shape) + str(y.shape))
    return x, y
