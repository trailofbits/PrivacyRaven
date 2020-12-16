import torch
import tqdm

from privacyraven.extraction.synthesis import process_data
from privacyraven.utils.query import get_target, query_model


def label_agreement(
    test_data,
    substitute_model,
    query_victim,
    victim_input_shape,
    substitute_input_shape,
):
    """Returns the number of agreed upon data points between victim and substitute,
    thereby measuring the fidelity of an extraction attack"""

    limit = int(len(test_data))
    if limit >= 100:
        limit = 100
    x_data, y_data = process_data(test_data, limit)

    substitute_result = get_target(substitute_model, x_data, substitute_input_shape)
    victim_result = query_victim(x_data)

    agreed = torch.sum(torch.eq(victim_result, substitute_result)).item()

    print(f"Out of {limit} data points, the models agreed upon {agreed}.")
    return agreed
