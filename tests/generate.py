"""
This is an example script of how to generate
boilerplate for hypothesis tests
"""

from hypothesis.extra import ghostwriter

import privacyraven.extraction.metrics as metrics

f = open("test_extraction_metrics.py", "w+")

f.write(ghostwriter.magic(metrics))
