import privacyraven.utils.query as query
from hypothesis.extra import ghostwriter

# print(ghostwriter.magic(synthesis), file='test_synthesis.py')

f = open("test_utils_query.py", "w+")

f.write(ghostwriter.magic(query))
