import torch
import tqdm
from privacyraven.utils.query import query_model, get_target
from privacyraven.extraction.synthesis import process_data


def label_agreement(
    test_data,
    substitute_model,
    query_victim,
    victim_input_shape,
    substitute_input_shape,
):
    """Returns the number of agreed upon data points between victim and substitute,
    thereby measuring the fidelity of an extraction attack"""

    l = int(len(test_data))
    x_data, y_data = process_data(test_data, l)

    substitute_result = get_target(substitute_model, x_data, substitute_input_shape)
    victim_result = query_victim(x_data)

    agreed = torch.sum(torch.eq(victim_result, substitute_result)).item()

    print(f"Out of {l} data points, the models agreed upon {agreed}.")
    return agreed
