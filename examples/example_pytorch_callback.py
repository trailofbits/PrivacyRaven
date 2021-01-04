"""
PrivacyRaven has support for user-defined callbacks, 
which may be passed as arguments into the currently available attack functions.

"""
import privacyraven as pr
from privacyraven.utils.data import get_emnist_data
from pl_bolts.callbacks import PrintTableMetricsCallback
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier
from pytorch_lightning.callbacks import Callback

emnist_train, emnist_test = get_emnist_data()


# Create a query function for a target PyTorch Lightning model
model = train_four_layer_mnist_victim(gpus=1)

def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))


# User-defined callback that inherits from the Pytorch's Callback class

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('End of epoch')

# Runs a Model Extraction Attack with the user-defined CustomCallback specified as an argument.
# See https://pytorch-lightning-bolts.readthedocs.io/en/latest/callbacks.html for all valid Pytorch
# Lightning callback hooks. Note that parentheses are needed while passing in the callback.

attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),
    victim_output_targets=10,
    substitute_input_shape=(3, 1, 28, 28),
    synthesizer="copycat",
    substitute_model_arch=FourLayerClassifier,
    substitute_input_size=784,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=1,
    callback=CustomCallback()
)

# Many built-in Pytorch callbacks are already very useful.  Consider the following example, which 
# runs the same Model Extraction Attack with the Pytorch built-in PrintTableMetricsCallback specified as an argument.
# After every epoch, a table should be displayed with all of the training metrics (e.g. training loss)
attack = ModelExtractionAttack(
    query=query_mnist,
    query_limit=100,
    victim_input_shape=(1, 28, 28, 1),
    victim_output_targets=10,
    substitute_input_shape=(3, 1, 28, 28),
    synthesizer="copycat",
    substitute_model_arch=FourLayerClassifier,
    substitute_input_size=784,
    seed_data_train=emnist_train,
    seed_data_test=emnist_test,
    gpus=1,
    callback=PrintTableMetricsCallback()
)

