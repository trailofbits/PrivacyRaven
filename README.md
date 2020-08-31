<p align="center">
 <img swidth="300" height="300" src="images/cropped.png">
</p>
<hr style="height:5px"/>

**PrivacyRaven** is a privacy testing library for deep learning systems.
You can use it to determine the susceptibility of a model to different privacy attacks; evaluate privacy preserving machine learning techniques; develop novel privacy metrics and attacks; and repurpose attacks for data provenance and other use cases.

PrivacyRaven supports label-only black-box model extraction, membership inference, and (soon) model inversion attacks.
We also plan to include differential privacy verification, automated hyperparameter optimization, more classes of attacks, and other features; see the GitHub issues for more information.
PrivacyRaven has been featured at the [OpenMined Privacy Conference](https://pricon.openmined.org/), [Empire Hacking](https://www.empirehacking.nyc/), [Trail of Bits blog](https://blog.trailofbits.com/), and other venues.

## Why use PrivacyRaven?

Deep learning systems, particularly neural networks, have proliferated in a wide range of applications, including privacy-sensitive use cases such as facial recognition and medical diagnoses.
However, these models are vulnerable to privacy attacks that target both the intellectual property of the model and the confidentiality of the training data.
Recent literature has seen an arms race between privacy attacks and defenses on various systems.
And until now, engineers and researchers have not had the privacy analysis tools they need to rival this trend.
Hence, we developed PrivacyRaven- a machine learning assurance tool that aims to be:
+ **Usable**: By providing multiple levels of abstraction, PrivacyRaven enables users to choose to automate much of the internal mechanics or directly control it when necessary based upon their use case and familiarity with the domain.
+ **Flexible**: A modular design makes these attack configurations customizable and interoperable. Furthermore, it allows new privacy metrics and attacks to be incorporated in a straightforward process.
+ **Efficient**: PrivacyRaven reduces the boilerplate, affording quick prototyping and fast experimentation. Each attack can be launched in less than 15 lines of code.

## How does it work?

### Demo

```python
#examples/extract_mnist.py
import privacyraven as pr
from privacyraven.utils.data import get_emnist_data
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import train_mnist_victim
from privacyraven.models.pytorch import ImagenetTransferLearning

# Create a query function for a PyTorch Lightning model
model = train_mnist_victim()

def query_mnist(input_data):
    return get_target(model, input_data)

# Obtain seed (or public) data to be used in extraction
emnist_train, emnist_test = get_emnist_data()

# Run a 'Knockoff Nets' Model Extraction Attack

attack = ModelExtractionAttack(
    query_mnist,
    100,
    (1, 28, 28, 1),
    10,
    (1, 3, 28, 28),
    "copycat",
    ImagenetTransferLearning,
    1000,
    emnist_train,
    emnist_test,
)
```

## Want to use PrivacyRaven?
1. Install [poetry](https://python-poetry.org/docs/).
2. Git clone this repository.
3. Run `poetry install`.

An official pip release is coming soon.

## Want to contribute to PrivacyRaven?

PrivacyRaven is still a work-in-progress. We invite you to contribute however you can whether you want to incorporate a new synthesis technique or make an attack function more readable. Please visit CONTRIBUTING.md to get started.

# References

While PrivacyRaven was built upon a plethora of research on attacking machine learning privacy, the research most critical to PrivacyRaven are:

+ [--]()
