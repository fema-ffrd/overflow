FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
RUN apt-get update && \
    apt-get install -y gdal-bin libgdal-dev build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system --no-cache --force-reinstall gdal[numpy]=="$(gdal-config --version).*"

COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN uv pip install --system .

# Set the entrypoint
ENTRYPOINT ["overflow"]
