# Installation

This guide covers different methods to install Overflow and its dependencies.

---

## Requirements

### System Requirements

Overflow is not comprehensively tested across all platforms. The recommended environment is:

- x86_64 or ARM64 architecture
- Linux operating system
- Python 3.11 or 3.12
- GDAL 3.12 with Python bindings and numpy support

Other platforms like macOS/Windows and other versions of Python/GDAL may work but are not officially tested and supported.

### Python Dependencies

The following Python packages are automatically installed:

- **NumPy** >= 1.26.4, <2
- **Numba** >= 0.59.0
- **Shapely** >= 2.0.6
- **Rich** >= 13.9.4
- **Click** >= 8.0.0

---

## Installation Methods

### Method 1: Micromamba (Recommended)

The recommended installation method uses conda/mamba to handle GDAL's installation.:

```bash
# Create a new environment with GDAL
micromamba create -n overflow python \
    gdal \
    -c conda-forge

# Activate the environment
conda activate overflow

# Install overflow from PyPI
pip install overflow-hydro
```

---

### Method 2: Docker (Easiest)

Docker provides a completely pre-configured environment with all dependencies:

```bash
# Pull the latest image
docker pull ghcr.io/fema-ffrd/overflow:latest

# Run overflow commands
docker run -it -v $(pwd)/data:/mnt/data ghcr.io/fema-ffrd/overflow:latest \
    pipeline \
    --dem_file /mnt/data/dem.tif \
    --output_dir /mnt/data/results \
    --chunk_size 512
```

See the Docker Usage Guide (TODO) for more details.

---

### Method 3: pip with System GDAL

If you have GDAL already installed on your system, you can install Overflow with pip or uv:

```bash
# Install overflow (requires system GDAL)
uv pip install overflow-hydro

# Optional: Install with GDAL Python bindings
uv pip install overflow-hydro[gdal]
```

!!! warning "Manual GDAL Installation Required"
    This method requires you to install GDAL system libraries manually. This can be complex on some platforms.

#### Installing GDAL System Libraries

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install gdal-bin libgdal-dev python3-gdal

    # Install overflow
    uv pip install overflow-hydro
    ```

=== "Brew (macOS)"

    ```bash
    brew install gdal

    # Install overflow
    uv pip install overflow-hydro
    ```

=== "Windows"

    ```bash
    # Use conda or OSGeo4W installer
    # https://trac.osgeo.org/osgeo4w/

    # Then install overflow
    uv pip install overflow-hydro
    ```

---

### Method 4: Development Installation

For contributing or running the latest development version:

```bash
# Clone the repository
git clone https://github.com/fema-ffrd/overflow.git
cd overflow

# Create conda environment with GDAL
conda create -n overflow-dev python gdal -c conda-forge
conda activate overflow-dev

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"

# Verify installation
pytest
```

Alternatively, you can use the devcontainer provided with the repository for a pre-configured development environment (recommended).

---

## Next Steps

TODO
