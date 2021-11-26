import attr
import pytorch_lightning as pl
from sklearn.neural_network import MLPClassifier
import torch
from torch.cuda import device_count
import copy
import sklearn.metrics as metrics
#from sklearn.metrics import roc_auc_score
from privacyraven.extraction.core import ModelExtractionAttack
#from privacyraven.membership_inf.robustness import find_robustness
from privacyraven.membership_inf.threshold import calculate_threshold_value
from privacyraven.utils.query import establish_query, get_target, query_model
from privacyraven.models.pytorch import ImagenetTransferLearning
import torchmetrics

@attr.s
class TransferMembershipInferenceAttack(object):
    """Launches a transfer-based membership inference attack"""
    gpu_availability = torch.cuda.device_count()
    data_point = attr.ib()
    query = attr.ib()
    query_limit = attr.ib(default=100)
    victim_input_shape = attr.ib(default=None)
    victim_output_targets = attr.ib(default=None)
    substitute_input_shape = attr.ib(default=None)
    synthesizer = attr.ib(default="copycat")
    substitute_model_arch = attr.ib(default=ImagenetTransferLearning)
    substitute_input_size = attr.ib(default=1000)
    seed_data_train = attr.ib(default=None)
    seed_data_test = attr.ib(default=None)
    threshold = attr.ib(default=None)
    # test_data = attr.ib(default=None)

    transform = attr.ib(default=None)
    batch_size = attr.ib(default=100)
    num_workers = attr.ib(default=4)
    gpus = attr.ib(default=gpu_availability)
    max_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=1e-3)
    art_model = attr.ib(default=None)
    callback = attr.ib(default=None)
    trainer_args = attr.ib(default=None)

    extraction_attack = attr.ib(init=False)
    substitute_model = attr.ib(init=False)
    query_substitute = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.query = establish_query(self.query, self.victim_input_shape)

        # We use the dict of the attack to unpack all the extraction arguments
        # This will need to be changed as ModelExtractionAttack is changed

        config = attr.asdict(self)
        extract_args = copy.deepcopy(config)
        # print(extract_args)
        extract_args.pop("data_point")
        extract_args.pop("threshold")
        extract_args = extract_args.values()

        self.extraction_attack = ModelExtractionAttack(*extract_args)
        self.substitute_model = extraction.substitute_model

        self.query_substitute = lambda x: query_model(substitute, x, self.substitute_input_shape)
        pred, target = query_substitute(self.data_point)

        # target = target.unsqueeze(0)
        # output = torch.nn.functional.cross_entropy(pred, target)

        # t_pred, t_target = query_substitute(self.seed_data_train)

        # We need diff formats for threshold: #, function, string (?) 

        # threshold = torch.nn.functional.cross_entropy()
        # print("Cross Entropy Loss is: " + output)
        # print("AUROC is: " + auroc)

        # We need multiple: binary classifier & threshold 
        # This maps to attackNN-based and metric-based attacks
        if threshold = None:
            binary_classifier = True
        else:
            binary_classifier = False
            tr = calculate_threshold_value(threshold)

        # Threshold value must be number, string in list of functions, OR
        # function

