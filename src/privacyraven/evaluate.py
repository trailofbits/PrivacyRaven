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
    agreed = 0

    for i in tqdm(range(1, points)):
        # Case: Torchvision Dataset
        x, y = test[i]

        substitute_result = int(query(substitute_model, x, substitute_size))
        victim_result = int(query(victim_model, x, victim_size))

        if substitute_result == victim_result:
            agreed = agreed + 1

    print(
        "Out of "
        + str(points)
        + " data points, the models agreed upon "
        + str(agreed)
        + "."
    )

    return agreed
