from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import FourLayerClassifier
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack

from torch import topk, add, log as vlog


def get_target(model, input_data, input_size=None):
    """Returns the predicted target of a Pytorch model
    Parameters:
        model: A pl.LightningModule or Torch module to be queried
        input_data: A Torch tensor entering the model
        input_size: A tuple of ints describes the shape of x
    Returns:
        target: An Torch tensor displaying the predicted target"""
    prediction, target = query_model(model, input_data, input_size)
    return target


# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    prediction, target = query_model(model, input_data, (1, 28, 28, 1))
    return prediction

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


    # Temporary model to be passed into Model Extraction
    m = train_four_layer_mnist_victim(
        gpus=1, 
        input_size = input_size, 
        output_size = output_size
    )

    # Classifier to assign prediction vector and target based on (E)MNIST training data
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
    )


    # This is nowhere near complete but 
    # The idea here is that we query the model each time 
    for k in range(len(dataset_train)):

        # Fwx is the training vector outputted by our model Fw
        Fwx = add(vlog(query_mnist(input_data)), c)

        # Let Fw_t denote the truncated vector
        Fwx_t = topk(t, Fwx)


    


if __name__ == "__main__":
    emnist_train, emnist_test = get_emnist_data()

    joint_train_inversion_model(dataset_train=emnist_train)