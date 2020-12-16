"""
This is an example script of how to generate
boilerplate for hypothesis tests
"""

import privacyraven.extraction.metrics as metrics
from hypothesis.extra import ghostwriter

f = open("test_extraction_metrics.py", "w+")

f.write(ghostwriter.magic(metrics))
