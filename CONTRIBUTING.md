<p align="center">
 <img swidth="300" height="300" src="images/cropped.png">
</p>
<hr style="height:5px"/>

Thank you for considering contributing to PrivacyRaven!
Feel free to ask any questions if something is unclear.
We’ve codified a set of guidelines and instructions to make contributing to PrivacyRaven as easy as possible.
Please note that these guidelines are not rigid and can be broken if necessary.

+ Build instructions are contained in the README. Fork your own repository from GitHub and rebase your fork with new changes from master.
+ Raise or be assigned to an issue before submitting a pull request. After submitting a pull request, one of our core maintainers must approve it before it is pushed.
+ When reporting a bug, provide as much information as possible with regards to your setup.
+ When suggesting a feature, feel free to leave it open-ended. Provide links to similar implementations or reference papers as applicable.
+ Ensure that the code you’ve contributed passes all tests and formatting checks. PrivacyRaven uses Python black, isort, flake8, and nox.
+ Be generous with comments and explanations
+ Create tests and documentation for your contribution
+ For now, documentation merely entails incorporating docstrings like so:
```python

def example(a):
   """Does something

   Parameters:
      a: data type that represents something

   Returns:
      a data type that represents something else"""

class another(object):
    """Does another thing

    Attributes:
       a: data type that represents something else"""
```
+ Clearly disclose all known limitations
+ Center the user while developing. Simplify the API as much as possible
+ Make sure that a user can quickly understand what your code does even without understanding how it works. Reference the [Python API Checklist](https://github.com/vintasoftware/python-api-checklist/blob/master/checklist-en.md) to confirm usability.
+ Build from the fundamental building blocks.
+ To add a new attack in `attack.py`, make sure to build functions for any novel synthesizer, robustness metrics, or subset sampling strategies, and that the new attack is included in the run-all attacks feature.
+ To add a new synthesizer for `synthesis.py` or robustness metric for `robust.py`, maintain the same function signature.
+ When building classes, use [attrs](https://www.attrs.org/en/stable/).
+ We prefer data-specific code to be written in PyTorch and generally adhere to the [PyTorch Style Guide](https://github.com/IgorSusmelj/pytorch-styleguide).
