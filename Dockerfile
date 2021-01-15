FROM python:3.6

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

ADD . ~/PrivacyRaven
WORKDIR ~/PrivacyRaven

ENV PATH="${PATH}:/root/.poetry/bin"

RUN poetry install
#WORKDIR examples/

#RUN poetry run pip install fsspec
#RUN poetry run python create_synthesizer.py
#RUN poetry run python example_pytorch_callback.py

#WORKDIR ../

RUN poetry run pip install nox
RUN poetry update
RUN pip install nox
RUN nox
