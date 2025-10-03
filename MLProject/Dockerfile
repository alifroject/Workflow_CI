FROM continuumio/miniconda3

WORKDIR /app


COPY conda.yaml .
RUN conda env create -f conda.yaml


SHELL ["conda", "run", "-n", "ml-env", "/bin/bash", "-c"]


COPY . .


CMD ["conda", "run", "--no-capture-output", "-n", "ml-env", "python", "modelling.py"]
