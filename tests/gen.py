import privacyraven.extraction.synthesis as synthesis
from hypothesis.extra import ghostwriter

# print(ghostwriter.magic(synthesis), file='test_synthesis.py')

f = open("test_synthesis.py", "w+")

f.write(ghostwriter.magic(synthesis))
