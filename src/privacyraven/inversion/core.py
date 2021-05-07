import torch.nn.functional as nnf
from tqdm import tqdm
from torch.cuda import device_count
from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import *
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import query_model, get_target
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.extraction.synthesis import process_data
from privacyraven.utils.model_creation import  NewDataset
from torch import topk, add, log as vlog, tensor, sort

# Create a query function for a target PyTorch Lightning model
def get_prediction(model, input_data, emnist_dimensions=(1, 28, 28, 1)):
    # PrivacyRaven provides built-in query functions
    prediction, target = query_model(model, input_data, emnist_dimensions)
    return prediction

# Relabels (E)MNIST data via the mapping (784, 10) -> (10, 784)
# Takes in a tensor of all
def relabel_emnist_data(img_tensor, Fwx_t_tensor):
    return NewDataset(Fwx_t_tensor.float(), img_tensor.long())

def trunc(k, v):

    # kth smallest element
    b = sorted(v)[-k - 1]
    nonzero = 0

    for (i, vi) in enumerate(v):
        if vi < b or (vi != 0 and nonzero > k): v[i] = 0    
        nonzero += 1

    return v
    

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


    dataset_len = len(dataset_train)
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
    ).substitute_model

    # This is nowhere near complete but 
    # The idea here is that we query the model each time 

    forward_model.eval()

    training_batch = torch.Tensor(dataset_len, 28, 28)
    labels = torch.Tensor(dataset_len, 10)
   

    # We use NewDataset to synthesize the training and test data, to ensure compatibility with Pytorch Lightning NNs.
    device = "cuda:0" if device_count() else None


    relabeled_data = relabel_emnist_data(emnist_train.data[:dataset_len], labels)
    
    # Intermediate tensor dimensions are (2, 10)
    inversion_model = train_four_layer_mnist_inversion(
        gpus=1,
        datapoints = relabeled_data,
        rand_split_val=[100, 50, 50]
    )

    return inversion_model
    
if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    joint_train_inversion_model(
        dataset_train=emnist_train,
        dataset_test=emnist_test
    )