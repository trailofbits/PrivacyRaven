import torch
import tqdm

from privacyraven.query import get_target


def label_agreement(
    test,
    substitute_model,
    victim_model,
    query=get_target,
    substitute_size=None,
    victim_size=None,
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

    for i in tqdm(range(1, points)):
        # Case: Torchvision Dataset
        x, y = test[i]

        substitute_result = int(query(substitute_model, x, substitute_size))
        victim_result = int(query(victim_model, x, victim_size))

        if substitute_result == victim_result:
            agreed = agreed + 1

    print(f"Out of {points} data points, the models agreed upon {agreed}.")
    return agreed
