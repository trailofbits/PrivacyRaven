import attr
import pytorch_lightning as pl


@attr.s
class MembershipInferenceAttack:
    query = attr.ib()
    query_limit = attr.ib()
    victim_input_shape = attr.ib()
    victim_output_targets = attr.ib()
    substitute_input_shape = attr.ib()
