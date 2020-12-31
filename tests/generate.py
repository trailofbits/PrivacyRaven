"""
This is an example script of how to generate
boilerplate for hypothesis tests
"""

from hypothesis.extra import ghostwriter
import privacyraven.extraction.metrics as metrics

# The above line imports the source code

file_name = "test_this_code.py"

f = open(file_name, "w+")

f.write(ghostwriter.magic(metrics))
