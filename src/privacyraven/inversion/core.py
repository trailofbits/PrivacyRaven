import torch.nn.functional as nnf
from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import FourLayerClassifier
from privacyraven.utils.data import get_emnist_data
from privacyraven.utils.query import query_model
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.extraction.synthesis import process_data
from torch import topk, add, log as vlog, tensor


# Create a query function for a target PyTorch Lightning model
def query_mnist(model, input_data, emnist_dimensions=(1, 28, 28, 1)):
    # PrivacyRaven provides built-in query functions
    prediction, target = query_model(model, input_data, emnist_dimensions)
    return prediction

# Relabels (E)MNIST data via the mapping (784, 10) -> (10, 784)
def relabel_emnist_datapoint(img, Fwx_t):
    return (Fwx_t, img)

# Trains the forward and inversion models
def joint_train_inversion_model(
    input_size = 784,
    output_size = 10,
    dataset_train = None,
    dataset_test = None,
    data_dimensions = (1, 28, 28, 1),
    t = 2,
    c = 10,
    ):
    
    # The following is a proof of concept of Figure 4 from the paper
    # "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
    # We first train a classifier on a dataset to output a prediction vector 


    # Classifier to assign prediction vector and target based on (E)MNIST training data
    temp_model = train_four_layer_mnist_victim(
        gpus=1, 
        input_size = input_size, 
        output_size = output_size
    )

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
    ).get_substitute_model()

    # This is nowhere near complete but 
    # The idea here is that we query the model each time 

    training_batch = torch.Tensor(0, 28, 28)

    for k in range(len(dataset_train)):

        # (image tensor, label tensor)
        img_tensor, label = process_data(emnist_train)

        # Fwx is the training vector outputted by our model Fw
        # We couple 
        Fwx = add(vlog(torch.nnf(query_mnist(img_tensor), dim=1), c))
        
        # Let Fw_t denote the truncated vector with the top k highest classes.
        Fwx_t = topk(t, Fwx)

        relabeled_datapoint = relabel_emnist_datapoint(img_tensor, Fwx_t)

        training_batch.cat(relabeled_datapoint)

    inversion_model = train_four_layer_mnist_inversion(
            gpus=1,
            input_size = input_size,
            output_size = output_size
    )

    return inversion_model
    


if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    joint_train_inversion_model(
        dataset_train=emnist_train,
        dataset_test=emnist_test
    )

    # 

