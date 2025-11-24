# Overflow

[![Tests](https://github.com/fema-ffrd/overflow/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/fema-ffrd/overflow/actions/workflows/lint_and_test.yml)
[![Documentation](https://github.com/fema-ffrd/overflow/actions/workflows/docs.yml/badge.svg)](https://fema-ffrd.github.io/overflow/)
[![PyPI](https://img.shields.io/pypi/v/overflow-hydro)](https://pypi.org/project/overflow-hydro/)
[![Python Version](https://img.shields.io/pypi/pyversions/overflow-hydro)](https://pypi.org/project/overflow-hydro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Warning**: This software is currently in alpha status. While functional, it may contain bugs and undergo breaking changes. See the [documentation](https://fema-ffrd.github.io/overflow/) for more details.

Overflow is a high-performance Python library for hydrological terrain analysis that specializes in processing massive Digital Elevation Models (DEMs) through parallel, tiled algorithms.

## Overview

Overflow provides a complete toolchain for extracting hydrographic features from raw elevation data:

- **Terrain Conditioning**: Breach and fill depressions in DEMs
- **Flow Routing**: Calculate D8 flow direction with flat resolution and flow accumulation
- **Feature Extraction**: Extract stream networks, delineate basins, and compute flow lengths

Unlike traditional GIS tools that rely on virtual memory, Overflow uses sophisticated tiled algorithms with parallel processing to handle datasets of any size efficiently.

## Key Features

- **Highly Parallel**: Every algorithm designed for parallel execution using Numba.
- **Memory Efficient**: Process DEMs larger than RAM through tiled algorithms with efficient I/O patterns.
- **Scalable**: Fixed number of passes over data regardless of size. Every operation is designed to scale from laptop-sized datasets to continental DEMs without requiring specialized hardware.
- **Accurate**: Maintains correctness across tile boundaries using graph-based approaches. The algorithms produce results that are equivalent to authoritative methods used throughout the hydrological community.
- **Modern Algorithms**: Implements state-of-the-art priority-flood filling, least-cost breaching, and flat resolution algorithms

## Quick Start

### Installation

```bash
# Create conda environment with system dependencies
conda create -n overflow python gdal -c conda-forge
conda activate overflow

# Install overflow from PyPI
pip install overflow-hydro
```

### Basic Usage

#### Python API

```python
import overflow

# Process a DEM through the complete pipeline
overflow.breach("input.tif", "breached.tif")
overflow.fill("breached.tif", "filled.tif")
overflow.flow_direction("filled.tif", "flowdir.tif")
overflow.accumulation("flowdir.tif", "flowacc.tif")
overflow.streams("flowacc.tif", "flowdir.tif", "streams/")
overflow.basins("flowdir.tif", "basins.tif")
```

#### Command Line Interface

```bash
# Run the complete pipeline
overflow pipeline \
    --dem_file input.tif \
    --output_dir ./results

# Or run individual steps
overflow breach input.tif breached.tif
overflow fill breached.tif filled.tif
overflow flow-direction filled.tif flowdir.tif
overflow accumulation flowdir.tif flowacc.tif
```

### Docker

```bash
docker pull ghcr.io/fema-ffrd/overflow:latest

docker run -v $(pwd):/data ghcr.io/fema-ffrd/overflow:latest \
    pipeline --dem_file /data/input.tif --output_dir /data/results
```

## Documentation

For detailed documentation, algorithm descriptions, and API reference, visit:

**[https://fema-ffrd.github.io/overflow/](https://fema-ffrd.github.io/overflow/)**

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://fema-ffrd.github.io/overflow/](https://fema-ffrd.github.io/overflow/)
- **Issues**: [GitHub Issues](https://github.com/fema-ffrd/overflow/issues)
