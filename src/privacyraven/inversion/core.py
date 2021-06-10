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

# Trains the forward and inversion models
def joint_train_inversion_model(
    input_size = 784,
    output_size = 784,
    dataset_train = None,
    dataset_test = None,
    data_dimensions = (1, 28, 28, 1),
    t = 5,
    c = 7,
    plot=False
    ):
    
    # The following is a proof of concept of Figure 4 from the paper
    # "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
    # We first train a classifier on a dataset to output a prediction vector 


    dataset_len = 200
    # Classifier to assign prediction vector and target based on (E)MNIST training data
    temp_model = train_four_layer_mnist_victim(
        gpus=1
    )

    def query_mnist(input_data):
        # PrivacyRaven provides built-in query functions
        return get_target(temp_model, input_data, (1, 28, 28, 1))


    forward_model = ModelExtractionAttack(
        query_mnist,
        200,  # Less than the number of MNIST data points: 60000
        (1, 28, 28, 1),
        10,
        (3, 1, 28, 28),  # Shape of an EMNIST data point
        "copycat", # "copycat",
        FourLayerClassifier,
        784,  # 28 * 28 or the size of a single image
        emnist_train,
        emnist_test,
        gpus=1
    ).substitute_model

    # Due to PrivacyRaven's black box threat model, we first run a model extraction attack on the
    # target classifier to extract and train a fully-trained substitute model, which the user has white-box access to.
    # Ideally, if the model extraction is successful, then this substitute model should approximate the target classifier
    # to a reasonable degree of fidelity and accuracy.
    # We then train the inversion model using the substitute model to query the auxiliary dataset on
    # under the objective of minimizing the MSE loss between the reconstructed and auxiliary
    # datapoints.  


    prediction = get_prediction(forward_model, emnist_train[0][0].float())

    if plot:
        plt.imshow(emnist_train[0][0][0].reshape(28, 28), cmap="gray")
        plt.show()
        print("Prediction: ", prediction, prediction.size())


    # Inversion training process occurs here
    
    inversion_model = train_mnist_inversion(
        forward_model,
        gpus=1,
        forward_model=forward_model,
        inversion_params={"nz": 10, "ngf": 128, "affine_shift": c, "truncate": t},
        max_epochs=300,
    )

    reconstructed = inversion_model(prediction[0]).to("cpu")
    plt.imshow(reconstructed[0][0].reshape(32, 32), cmap="gray")
    plt.show()
    return inversion_model
    
if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    model = joint_train_inversion_model(
        dataset_train=emnist_train,
        dataset_test=emnist_test,
    )