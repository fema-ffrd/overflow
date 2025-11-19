FROM ghcr.io/osgeo/gdal:ubuntu-small-3.12.0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

RUN uv venv $UV_PROJECT_ENVIRONMENT --system-site-packages

WORKDIR /app

COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN uv pip install .

# Set the entrypoint
ENTRYPOINT ["overflow"]

LABEL description="High-performance hydrological terrain analysis"
LABEL displayName="Overflow"