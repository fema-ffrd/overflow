# Overview

Overflow is a high-performance Python library designed for processing Digital Elevation Models (DEMs) at scale. The library addresses the computational challenges of hydrological terrain analysis when working with massive raster datasets that can range from local watersheds to continental-scale terrain models.

## What Overflow Does

Overflow provides a complete suite of tools for deriving hydrographic features from raw elevation data. The library handles the full processing pipeline from terrain conditioning through feature extraction. This includes removing artifacts and depressions in the elevation data, computing how water flows across the landscape, calculating drainage area, and extracting vector representations of stream networks and basin boundaries.

## Who Should Use Overflow

Overflow is designed for anyone who needs to run these hydrolocial process on large DEMs efficiently. The library is particularly valuable when working with datasets that exceed the practical limits of traditional single-threaded tools or when processing time is a critical constraint.

Users should have basic familiarity with hydrological concepts such as flow direction, flow accumulation, and drainage networks. The library assumes you are comfortable working with geospatial raster data and understand fundamental GIS concepts. Programming experience with Python is helpful for using the API, though the command-line interface provides access to core functionality without requiring code.

## Design Philosophy

Overflow prioritizes scalability and performance while maintaining correctness. The algorithms make practical trade-offs that enable efficient processing of massive datasets. Most of the algorithms produce results that are mathematically equivalent to authoritative methods used throughout the hydrological community.

## The Scaling Problem

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
