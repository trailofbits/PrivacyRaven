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
    x_data, y_data = process_data(test_data, 100)

    substitute_result = get_target(substitute_model, x_data, substitute_input_shape)
    victim_result = query_victim(x_data)

    agreed = torch.sum(torch.eq(victim_result, substitute_result)).item()

    print(f"Out of 100 data points, the models agreed upon {agreed}.")
    return agreed
