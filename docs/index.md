---
title: Overflow
subtitle: Hydrological Terrain Analysis for Massive DEMs
---

# Overflow

!!! warning "Alpha Software"
    This software is currently in **alpha** status. While functional, it may contain bugs, have incomplete features, and undergo breaking changes. Use in production environments at your own risk. We welcome feedback and bug reports on our [GitHub Issues](https://github.com/fema-ffrd/overflow/issues) page.

Overflow specializes in processing massive Digital Elevation Models (DEMs) through parallel, tiled algorithms.

---

## Why Overflow?

Traditional hydrological tools are often single-threded and rely on virtual-memory tiling (swapping memory to disk) to handle large datasets. This creates inefficient I/O patterns with no garuntees on how many times the same tile may be swapped into and out of memory. Given the wide availability of multi-core processors in modern computers and the increasing size of raster datasets used for hydrological analysis, existing tools are not well-suited for efficient processing of large-scale terrain data today. Overflow was developed to provide a modern solution for large-scale hydrological terrain analysis.

Every workflow in Overflow is designed for parallel execution using Numba JIT-compiled algorithms. This is achieved through state-of-the-art tiled, topological approaches that implement efficient IO access patterns even for global operations. Processes in Overflow are guaranteed to complete in a fixed number of passes over the data and create results identical to the authoritative widely used algorithms, regardless of the dataset size.

## Scaling Global Algorithms

Raster processing algorithms vary significantly in complexity. While most tools can easily parallelize simple tasks, Overflow is specifically engineered to handle the "hard" problems in hydrology. These can be broadly categorized into three classes based on their computational complexity and data dependencies:

### Local Operations

Mathematically, a local operation maps a single input value to a single output value using a function $f$. There are no spatial dependencies. If $I$ is the input raster and $O$ is the output raster, then for each pixel at row $i$ and column $j$:

$$
O_{i,j} = f(I_{i,j})
$$

These operations consider each pixel in isolation. They are computationally inexpensive and trivial to parallelize because no information needs to be shared between pixels. These include operations like reclassifying elevation ranges (e.g., converting meters to feet) or calculating NDVI from imagery.

### Focal / Regional Operations

These operations calculate a value based on a pixel and its immediate neighbors (e.g., a 3x3 window). Parallelizing these requires "halo" or buffer regions around tiles to ensure edge pixels can see their neighbors in adjacent tiles. These include operations like calculating flow direction, slope, or local relief. Mathematically, a focal operation can be expressed as:

$$
O_{i,j} = f\left( \{ I_{u,v} \mid u \in [i-k, i+k], v \in [j-k, j+k] \} \right)
$$

Where $k$ defines the size of the neighborhood.

### Global Operations

These are the most computationally difficult algorithms to parallelize because the value of a single pixel can depend on **any other pixel** in the entire raster, regardless of distance. A change in elevation at one edge of the map can affect flow accumulation at the opposite edge. Unfortunately, these global operations include most of the useful hydrological processes. Flow accumulation, depression filling, undefined flow direction resolution, basin delineation, longest flow paths and stream network extraction are all global operations.

Most global algorithms in hydrology are defined recursively, making them inherently sequential. The value at a pixel depends on the topology of the flow network.

$$
O_{i,j} = f\left( \{ I_{u,v} \mid (u,v) \in \mathcal{P}_{(i,j)} \} \right)
$$

Where $P(i,j)$​ is the set of all pixels forming a directed path ending at $(i,j)$.

You cannot compute $O_{i,j}$ without resolving the state of pixels potentially located at the opposite end of the raster. This necessitates a graph-based approach. Overflow solves these global problems by processing data tile-by-tile to construct a connectivity graph, followed by a finalization pass that resolves dependencies across the entire domain.

## Core Hydrological Processes

Overflow provides a complete toolchain for deriving hydrographic features from raw elevation data. The core algorithms implemented in Overflow include:

Terrain Conditioning:

* Breaching
* Filling

Flow Routing:

* D8 Flow Direction with Flat/Undefined Flow Resolution
* Flow Accumulation

Feature Extraction:

* Stream Network Extraction
* Basin Delineation
* Upstream Flow Length with Longest Flow Path Extraction


### Python API

Overflow is available as a standard Python package making it accessible for integration into geospatial workflows leveraging Python's ecosystem.

```python
import overflow

overflow.breach(...)
overflow.fill(...)
overflow.flow_direction(...)
overflow.accumulation(...)
overflow.streams(...)
overflow.basins(...)
overflow.flow_length(...)
```

### Command Line Interface (CLI)

For batch processing and server environments, Overflow provides a robust CLI. It supports all core algorithms and provides a single command to run the complete processing pipeline end-to-end.

Available commands:

* `overflow breach` - Breach pits using least-cost paths
* `overflow fill` - Fill depressions in a DEM
* `overflow flow-direction` - Compute D8 flow directions (with flat resolution)
* `overflow accumulation` - Calculate flow accumulation
* `overflow streams` - Extract stream networks
* `overflow basins` - Delineate drainage basins
* `overflow flow-length` - Calculate upstream flow length (longest flow path)
* `overflow pipeline` - Run complete DEM processing workflow

The `pipeline` command orchestrates the full sequence of conditioning, flow direction, accumulation, and vector extraction in a single execution.

```bash
overflow pipeline \
    --dem_file raw_dem.tif \
    --output_dir ./results
```

### Docker Support

To ensure reproducibility and eliminate complex dependency management (specifically GDAL and system libraries), Overflow is distributed with an official Docker image.

The image is built on top of the official `osgeo/gdal` Ubuntu-based images, ensuring binary compatibility with the underlying geospatial libraries.


