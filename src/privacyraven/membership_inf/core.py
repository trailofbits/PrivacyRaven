import attr
import pytorch_lightning as pl
from sklearn.neural_network import MLPClassifier
import torch
from torch.cuda import device_count
import copy
from sklearn.metrics import roc_auc_score
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.membership_inf.robustness import find_robustness
from privacyraven.utils.query import establish_query, get_target, query_model
from privacyraven.models.pytorch import ImagenetTransferLearning


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
    test_data = attr.ib(default=None)

    transform = attr.ib(default=None)
    batch_size = attr.ib(default=100)
    num_workers = attr.ib(default=4)
    gpus = attr.ib(default=gpu_availability)
    max_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=1e-3)
    art_model = attr.ib(default=None)
    callback = attr.ib(default=None)
    trainer_args = attr.ib(default=None)


    def __attrs_post_init__(self):
        self.query = establish_query(self.query, self.victim_input_shape)

        # We use the dict of the attack to unpack all the extraction arguments
        # This will need to be changed as ModelExtractionAttack is changed

        config = attr.asdict(self)
        extract_args = copy.deepcopy(config)
        extract_args.pop("data_point")
        extract_args = extract_args.values()

        extraction = ModelExtractionAttack(*extract_args)
        substitute = extraction.substitute_model

        query_substitute = lambda x: query_model(substitute, x, self.substitute_input_shape)
        pred, target = query_substitute(self.data_point)
        # import pdb; pdb.set_trace()
        """
        print("Generating")
        print(pred.shape)
        print(target.shape)
        print("Using loss")
        target = torch.nn.functional.one_hot(target, self.victim_output_targets)

        loss = torch.nn.BCELoss()
        pred = torch.transpose(pred, 0, 1)
        target = target.unsqueeze(1)

        pred = pred.float().cpu()
        target = target.float().cpu()
        print(pred.shape)
        print(target.shape)
        output = loss(pred, target)
        """
        #loss = torch.nn.Cross

        # output = log_loss(target, pred)

        print(pred.shape)
        target = target.unsqueeze(0)
        # target = torch.nn.functional.one_hot(target, self.victim_output_targets)
        print(target.shape)
        #loss = torch.nn.functional.cross_entropy()
        output = torch.nn.functional.cross_entropy(pred, target)
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        one_hot_target = torch.nn.functional.one_hot(target,
                                                     self.victim_output_targets)
        import pdb; pdb.set_trace()

        auroc = roc_auc_score(target.cpu(), softmax_pred.cpu(), average='macro',
                              multi_class='ovr')

        # auroc = pl.metrics.functional.multiclass_auroc(softmax_pred, target, 10)
        #import pdb; pdb.set_trace()
        #auroc = pl.metrics.functional.classification.multiclass_auroc(torch.nn.functional.log_softmax(pred, dim=0), target, self.victim_output_targets)
        #auroc = calculate_auroc(pred, target)
        print("Cross Entropy Loss is: " + output)
        print("AUROC is: " + auroc)


