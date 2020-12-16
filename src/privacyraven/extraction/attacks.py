from functools import partial

import attr

from privacyraven.extraction.core import ModelExtractionAttack


def get_extraction_attrs():
    """Returns all the attributes of a Model Extraction Attack"""
    attributes = ModelExtractionAttack.__dict__["__attrs_attrs__"]
    attr_names = (a.name for a in attributes)
    return attr_names


def copycat_attack(*args, **kwargs):
    """Runs the CopyCat model extraction attack

    Arxiv Paper: https://arxiv.org/abs/1806.05476

    Presently, this function excludes subset sampling strategies"""
    copy = partial(ModelExtractionAttack, synthesizer=copycat_attack)
    return copy(*args, **kwargs)


def cloudleak(*args, **kwargs):
    """Runs CloudLeak model extraction attacks

    Returns an array of attacks that use synthesis functions
    based on adversarial/evasion attacks

    Based upon: https://bit.ly/31Npbgj

    Unlike the paper, this function does not include subset
    sampling strategies and relies upon different evasion
    attacks in order to comply with the threat model"""

    adv_synths = ["HopSkipJump"]
    results = []
    for s in adv_synths:
        attack = partial(ModelExtractionAttack, synthesizer=s)
        result = attack(*args, **kwargs)
        results = results.append(result)
    return results
