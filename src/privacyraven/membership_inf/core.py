import attr
import pytorch_lightning as pl
from sklearn.neural_network import MLPClassifier

from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.membership_inf.robustness import find_robustness


@attr.s
class MembershipInferenceAttack(object):
    """Launches a membership inference attack"""

    query = attr.ib()
    query_limit = attr.ib(default=100)
    victim_input_shape = attr.ib(default=None)
    victim_output_targets = attr.ib(default=None)
    substitute_input_shape = attr.ib(default=None)
    synthesizer = attr.ib(default="copycat")
    substitute_model = attr.ib(default=None)
    substitute_input_size = attr.ib(default=1000)
    seed_data_train = attr.ib(default=None)
    seed_data_test = attr.ib(default=None)
    transform = attr.ib(default=None)
    batch_size = attr.ib(default=100)
    num_workers = attr.ib(default=4)
    gpus = attr.ib(default=1)
    max_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=1e-3)
    extracted_model = attr.ib(init=False)
    robust = attr.ib(default="mmembership_inf_hopskipjump")
    X = attr.ib(init=False)
    y = attr.ib(init=False)
    shadow = attr.ib(init=False)

    def __attrs_post_init__(self):
        extract = self.extract_substitute()
        self.extracted_model = extract.substitute_model
        self.X, self.y = self.calculate_robustness()
        self.shadow = self.train_shadow_model()

    def extract_substitute(self):
        extract = ModelExtractionAttack(
            self.query,
            self.query_limit,
            self.victim_input_shape,
            self.victim_output_targets,
            self.substitute_input_shape,
            self.synthesizer,
            self.substitute_model,
            self.substitute_input_size,
            self.seed_data_train,
            self.seed_data_test,
            self.transform,
            self.batch_size,
            self.num_workers,
            self.gpus,
            self.max_epochs,
            self.learning_rate,
        )
        return extract

    def calculate_robustness(self):
        return find_robustness(
            self.robust,
            self.extracted_model,
            self.query_limit,
            self.victim_input_shape,
            self.substitute_input_shape,
            self.victim_output_targets,
        )

    def train_shadow_model(self):
        return MLPClassifier(random_state=1, max_iter=300).fit(self.X, self.y)
