FROM mambaorg/micromamba:debian13-slim

COPY env.yaml /tmp/env.yaml

# install micromamba environment
RUN micromamba create -n overflow -f /tmp/env.yaml && \
    micromamba clean -a -y

COPY src /app/src
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini

WORKDIR /app

ENTRYPOINT ["micromamba", "run", "-n", "overflow", "python", "src/overflow_cli.py"]