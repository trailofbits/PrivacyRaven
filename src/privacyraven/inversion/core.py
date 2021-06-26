import os
import random
import torch.nn.functional as nnf
from tqdm import tqdm
from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import *
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import query_model, get_target
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.extraction.synthesis import process_data
from privacyraven.utils.model_creation import  NewDataset, set_hparams
import matplotlib.pyplot as plt

# Create a query function for a target PyTorch Lightning model
def get_prediction(model, input_data, emnist_dimensions=(1, 28, 28, 1)):
    # PrivacyRaven provides built-in query functions
    prediction, target = query_model(model, input_data, emnist_dimensions)
    return prediction

def save_inversion_results(
    image, 
    reconstructed, 
    plot=False, 
    label=None,
    save_dir="results",
    filename="recovered"
):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Generates subplots and plots the results
    plt.subplot(1, 2, 1)
    plt.imshow(image[0], cmap="gray")
    plt.title("Auxiliary set ({})".format(label))
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed[0][0].reshape(32, 32), cmap="gray")
    plt.title("Reconstructed")
    plt.savefig(f"{save_dir}/{filename}.png")

    if plot:
        plt.show()

# Trains the forward and inversion models
def joint_train_inversion_model(
    dataset_train=None,
    dataset_test=None,
    data_dimensions = (1, 28, 28, 1),
    max_epochs=None,
    gpus=1,
    t=5,
    c=50
):
    
    # The following is a proof of concept of Figure 4 from the paper
    # "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
    
    temp_model = train_four_layer_mnist_victim(
        gpus=gpus
    )

    def query_mnist(input_data):
        # PrivacyRaven provides built-in query functions
        return get_target(temp_model, input_data, (1, 28, 28, 1))


    forward_model = ModelExtractionAttack(
        query_mnist,
        1000,  # Less than the number of MNIST data points: 60000
        (1, 28, 28, 1),
        10,
        (3, 1, 28, 28),  # Shape of an EMNIST data point
        "copycat", # "copycat",
        FourLayerClassifier,
        784,  # 28 * 28 or the size of a single image
        emnist_train,
        emnist_test,
        gpus=gpus
    ).substitute_model

    # Due to PrivacyRaven's black box threat model, we first run a model extraction attack on the
    # target classifier to extract and train a fully-trained substitute model, which the user has white-box access to.
    # Ideally, if the model extraction is successful, then this substitute model should approximate the target classifier
    # to a reasonable degree of fidelity and accuracy.
    # We then train the inversion model using the substitute model to query the auxiliary dataset on
    # under the objective of minimizing the MSE loss between the reconstructed and auxiliary
    # datapoints.  

    inversion_model = train_mnist_inversion(
        gpus=gpus,
        forward_model=forward_model,
        inversion_params={"nz": 10, "ngf": 128, "affine_shift": c, "truncate": t},
        max_epochs=max_epochs,
        batch_size=100
    )

    return forward_model, inversion_model


def test_inversion_model(
    forward_model,
    inversion_model,
    image, 
    filename="recovered",
    save=True,
    label=None,
    debug=True
):
    prediction = get_prediction(forward_model, image.float())
    
    if debug:
        print("Prediction vector: ", prediction)

    # Inversion training process occurs here
    image = nnf.pad(input=image, pad=(2, 2, 2, 2), value=image[0][0][0])
    reconstructed = inversion_model(prediction[0]).to("cpu")

    if save:
        save_inversion_results(image, reconstructed, label=label, filename=filename)

    return nnf.mse_loss(image, reconstructed)

if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    forward_model, inversion_model = joint_train_inversion_model(
        dataset_train=emnist_train,
        dataset_test=emnist_test,
        gpus=1,
        max_epochs=300
    )

    num_test = 250
    idx_array = random.sample(range(len(emnist_test)), num_test)

    for idx in idx_array:
        image, label = emnist_test[idx]

        loss = test_inversion_model(
            forward_model,
            inversion_model,
            image,
            label=str(label),
            filename=f"recovered_{idx}",
            debug=False
        )