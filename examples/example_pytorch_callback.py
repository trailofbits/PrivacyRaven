"""
PrivacyRaven has support for user-defined Pytorch Lightning Bolt callbacks which may be passed as arguments 
into the currently available attack functions.

A callback is just a function (or class in this case) that gets passed in as an argument to another
function, which should execute the callback function or class at some point during its runtime. 

Users should refer to https://pytorch-lightning-bolts.readthedocs.io/en/latest/callbacks.html to construct 
Pytorch Lightning Bolt callbacks.
"""
import privacyraven as pr
from privacyraven.utils.data import get_emnist_data
from pl_bolts.callbacks import PrintTableMetricsCallback
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_four_layer_mnist_victim
from privacyraven.models.four_layer import FourLayerClassifier
from pytorch_lightning.callbacks import Callback


# Trains victim model
model = train_four_layer_mnist_victim(gpus=1)

# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))

emnist_train, emnist_test = get_emnist_data()

# Below is a user-defined callback that inherits from the Pytorch's Lightning Bolt Callback class.
# All it does is print "End of epoch" at the end of a training epoch.

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('End of epoch')

# Runs a Model Extraction Attack with the user-defined CustomCallback specified as an argument.
# Note that parentheses are needed while passing in the callback, since 
# Pytorch Lightning bolt callbacks are classes that need to be instantiated.

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

# Many built-in Pytorch Lightning Bolt callbacks are already very useful.  Consider the following example, which 
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

