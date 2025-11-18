FROM mambaorg/micromamba:debian13-slim

COPY env.yaml /tmp/env.yaml

# install micromamba environment
RUN micromamba create -n overflow -f /tmp/env.yaml && \
    micromamba clean -a -y

COPY src /app/src
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

WORKDIR /app

USER root

# Install the overflow package
RUN micromamba run -n overflow pip install -e .

ENTRYPOINT ["micromamba", "run", "-n", "overflow", "overflow"]