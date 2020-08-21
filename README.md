<p align="center">
 <img swidth="200" height="200" src="images/cropped.png">
</p>
<hr style="height:5px"/>

**PrivacyRaven** is a privacy testing library for deep learning systems.
You can use it to determine the susceptibility of a model to different privacy attacks; evaluate privacy preserving machine learning techniques; develop novel privacy metrics and attacks; and repurpose attacks for data provenance and other use cases.

PrivacyRaven supports label-only black-box model extraction, membership inference, and model inversion attacks.
We plan to include differential privacy verification, automated hyperparameter optimization, more classes of attacks, and other features; see the GitHub issues for more information.
PrivacyRaven has been featured at the [OpenMined Privacy Conference](https://pricon.openmined.org/), [Empire Hacking](https://www.empirehacking.nyc/), [Trail of Bits blog](https://blog.trailofbits.com/), and other venues.

## Why use PrivacyRaven?

Deep learning systems, particularly neural networks, have proliferated in a wide range of applications, including privacy-sensitive use cases such as facial recognition and medical diagnoses.
However, these models are vulnerable to privacy attacks that target both the intellectual property of the model and the confidentiality of the training data.
Recent literature has seen an arms race between privacy attacks and defenses on various systems.
And until now, engineers and researchers have not had the privacy analysis tools they need to rival this trend.
Hence, we developed PrivacyRaven- a machine learning assurance tool that aims to be:
+ **Usable**: PrivacyRaven provides multiple levels of abstraction. Users can choose to automate much of the internal mechanics or directly control it when necessary based upon their use case and familiarity with the domain.
+ **Flexible**: PrivacyRaven’s modular design makes these attack configurations customizable and interoperable. Furthermore, it allows new privacy metrics and attacks to be incorporated in a straightforward process.
+ **Efficient**: PrivacyRaven reduces the boilerplate, affording quick prototyping and fast experimentation. Each attack can be launched in less than 15 lines of code.

## How does it work?

### Demo

```python
#examples/extract_mnist.py
```

## Want to use PrivacyRaven?
1. Install [poetry](https://python-poetry.org/docs/).
2. Git clone this repository.
3. Run `poetry install`.

An official pip release is coming soon.

## Want to contribute to PrivacyRaven?
