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

# Relabels (E)MNIST data via the mapping (784, 10) -> (10, 784)
# Takes in a tensor of all
def generate_dataset(img_tensor, Fwx_t_tensor):
    return NewDataset(img_tensor.float(), Fwx_t_tensor.float())

# Trains the forward and inversion models
def joint_train_inversion_model(
    input_size = 784,
    output_size = 784,
    dataset_train = None,
    dataset_test = None,
    data_dimensions = (1, 28, 28, 1),
    t = 10,
    c = 10,
    ):
    
    # The following is a proof of concept of Figure 4 from the paper
    # "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
    # We first train a classifier on a dataset to output a prediction vector 


    dataset_len = 200
    # Classifier to assign prediction vector and target based on (E)MNIST training data
    temp_model = train_four_layer_mnist_victim(
        gpus=0
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
        gpus=0
    ).substitute_model

    # Due to PrivacyRaven's black box nature, we first run a model extraction attack 
    # Given a set of queries to a black box model.  From this, we construct a fully-trained substitute model
    # that approximates 
    # The idea here is that we query the model each time 

    forward_model.eval()

    # We use NewDataset to synthesize the training and test data, to ensure compatibility with Pytorch Lightning NNs.
    # print("EMNIST point 1: ", list(zip(*emnist_train))[0])
    relabeled_data = generate_dataset(emnist_train.data[:dataset_len], emnist_train.targets[:dataset_len])

    prediction = get_prediction(forward_model, emnist_train[0][0].float())
    plt.imshow(emnist_train[0][0][0].reshape(28, 28), cmap="gray")
    plt.show()
    print("Prediction: ", prediction, prediction.size())
    
    # Intermediate tensor dimensions are (1, 10)
    
    inversion_model = train_mnist_inversion(
        forward_model,
        gpus=0,
        datapoints=relabeled_data,
        forward_model=forward_model,
        inversion_params={"nz": 10, "ngf": 128, "affine_shift": 7, "truncate": 3}
    )

    reconstructed = inversion_model(prediction[0])
    plt.imshow(reconstructed[0][0].reshape(32, 32), cmap="gray")
    plt.show()
    return inversion_model
    
if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    model = joint_train_inversion_model(
        dataset_train=emnist_train,
        dataset_test=emnist_test
    )