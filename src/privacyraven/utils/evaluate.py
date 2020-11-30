import torch
import tqdm
from privacyraven.utils.query import query_model, get_target
from privacyraven.extraction.synthesis import process_data


def label_agreement(test_data, substitute_model, query_victim, victim_input_shape, substitute_input_shape):
    agreed = 0

    print(test_data.size())

    x_data, y_data = process_data(test_data, None)

    # for x in x_data:
        #substitute_result = int(query(substitute_model, x, substitute_shape))
        # victim_result = int(query(victim_model, x, victim_shape))

    substitute_result_tensor, substitute_result_targets = query_model(substitute_model, x_data, substitute_input_shape)
    victim_result_tensor, victim_result_targets = query_victim(x_data)

    import pdb; pdb.set_trace()

    print(f"Out of {points} data points, the models agreed upon {agreed}.")
    return agreed


def old_label_agreement(
    test_data,
    substitute_model,
    victim_model,
    query=get_target,
    substitute_shape=None,
    victim_shape=None,
    points=100,
):
    """Calculates the agreement between the victim and substitute models

    Parameters:
        test: A tuple-like dataset to sample test images from
        substitute_model: A model trained upon synthetic data
        victim_model: A model that has been extracted
        query: A function that returns labels of model given data
        substitute_size: A tuple with size of data sent to the
                         substitute model
        victim_size: A tuple with size of data sent to the victim
                     model
        points: An integer representing the number of test samples

    Returns:
        agreed: An integer of the number of samples with the same
                predicted labels

    """
    agreed = 0

    # for i in tqdm(range(1, points)):
    # Case: Torchvision Dataset
    # x, y = test[i]

    x_data, y_data = process_data(test_data)

    for x in x_data:
        substitute_result = int(query(substitute_model, x, substitute_shape))
        # victim_result = int(query(victim_model, x, victim_shape))

        victim_result = int(victim_model(x))

        if substitute_result == victim_result:
            agreed = agreed + 1

    print(f"Out of {points} data points, the models agreed upon {agreed}.")
    return agreed
