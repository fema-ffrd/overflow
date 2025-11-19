# Use a standard Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gdal-bin libgdal-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements files
COPY requirements.txt .

# Install the dependencies
RUN uv pip install --system -r requirements.txt

# Copy the application code
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

# Install the overflow package
RUN uv pip install --system .

# Set the entrypoint
ENTRYPOINT ["overflow"]
