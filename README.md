# PrivacyRaven

**PrivacyRaven** is a privacy testing library for deep learning systems.
You can use it to determine the susceptibility of a model to different privacy attacks; evaluate privacy preserving machine learning techniques; develop novel privacy metrics and attacks; and repurpose attacks for data provenance and other use cases.

## Why use PrivacyRaven?

Deep learning systems, particularly neural networks, have proliferated in a wide range of applications, including privacy-sensitive use cases such as facial recognition and medical diagnoses.
However, these models are vulnerable to privacy attacks that target both the intellectual property of the model and the confidentiality of the training data.
Recent literature has seen an arms race between privacy attacks and defenses on various systems.
And until now, engineers and researchers have not had the privacy analysis tools they need to rival this trend.
Hence, we developed PrivacyRaven: a comprehensive privacy testing suite for deep learning systems optimized for usability and efficiency.

## How does it work?

### Model Extraction Demo

```python
import privacyraven as pr
from privacyraven.data import get_emnist_data
from privacyraven.extraction import ModelExtractionAttack
from privacyraven.query import get_target
from privacyraven.victim import train_mnist_victim


# Create a query function for a Pytorch Lightning model
model = train_mnist_victim()


def query_mnist(input_data):
    return get_target(model, input_data)


# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a 'Knockoff Nets' Model Extraction Attack
test = ModelExtractionAttack(
    query_mnist,
    "knockoff",
    (1, 28, 28, 1),
    (1, 3, 28, 28),
    100,
    emnist_train,
    emnist_test,
)
```

## Want to use PrivacyRaven?
1. Install [poetry](https://python-poetry.org/docs/).
2. Git clone this repository.
3. Run `poetry install`.
