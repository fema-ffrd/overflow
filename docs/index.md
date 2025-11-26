---
title: Overflow
subtitle: Hydrological Terrain Analysis for Massive DEMs
hide:
  - navigation
  - toc
  - path
---

<div class="hero-section" style="text-align: left;">
  
  <div class="cli-banner" aria-label="Overflow Logo">
   ╔═╗╦  ╦╔═╗╦═╗╔═╗╦  ╔═╗╦ ╦
<span class="b-dim">░</span><span class="b-mid">▒</span><span class="b-bold">▓</span>║ ║╚╗╔╝║╣ ╠╦╝╠╣ ║  ║ ║║║║<span class="b-bold">▓</span><span class="b-mid">▒</span><span class="b-dim">░</span>
   ╚═╝ ╚╝ ╚═╝╩╚═╚  ╩═╝╚═╝╚╩╝
  </div>

  <p style="font-family: 'JetBrains Mono', monospace; color: var(--md-default-fg-color); font-size: 1.2rem; margin-top: 0; margin-bottom: 2rem; opacity: 0.8;">
    :: HYDROLOGICAL TERRAIN ANALYSIS
  </p>
  
  <div style="font-size: 1rem; max-width: 600px; line-height: 1.6; margin-bottom: 3rem;">
    A high-performance library for processing massive Digital Elevation Models.
  </div>

  <div class="grid-buttons">
    <a href="introduction/quickstart/" class="md-button md-button--primary">
      > QUICKSTART
    </a>
    <a href="getting-started/installation/" class="md-button">
      INSTALLATION
    </a>
  </div>
</div>

<hr style="border-bottom: 1px solid var(--border-color); margin: 3rem 0;">

!!! warning "Alpha Software"
    This software is currently in **alpha** status. While functional, it may contain bugs, have incomplete features, and undergo breaking changes. Use in production environments at your own risk. We welcome feedback and bug reports on our [GitHub Issues](https://github.com/fema-ffrd/overflow/issues) page.

---

## Why Overflow?

Traditional hydrological tools are often single-threaded and rely on virtual-memory tiling (swapping memory to disk) to handle large datasets. This creates inefficient I/O patterns with no garuntees on how many times the same tile may be swapped into and out of memory. Given the wide availability of multi-core processors in modern computers and the increasing size of raster datasets used for hydrological analysis, existing tools are not well-suited for efficient processing of large-scale terrain data today. Overflow was developed to provide a modern solution for large-scale hydrological terrain analysis.

Every workflow in Overflow is designed for parallel execution using Numba JIT-compiled algorithms. This is achieved through state-of-the-art tiled, topological approaches that implement efficient IO access patterns even for global operations. Processes in Overflow are guaranteed to complete in a fixed number of passes over the data regardless of the dataset size.

## Core Hydrological Processes

Overflow provides a complete toolchain for deriving hydrographic features from raw elevation data. The core algorithms implemented in Overflow include:

<div class="system-modules">

  <div class="module-group">
    <h4 class="module-header">01 // CONDITIONING</h4>
    <ul class="module-list">
      <li>
        <a href="../../user-guide/terrain-conditioning/breach/">Breach</a>
        <span class="specs">Carve paths through barriers</span>
      </li>
      <li>
        <a href="../../user-guide/terrain-conditioning/fill/">Fill</a>
        <span class="specs">Fill surface depressions</span>
      </li>
    </ul>
  </div>

  <div class="module-group">
    <h4 class="module-header">02 // ROUTING</h4>
    <ul class="module-list">
      <li>
        <a href="../../user-guide/flow-routing/flow-direction/">Flow Direction</a>
        <span class="specs">Determine steepest descent</span>
      </li>
      <li>
        <a href="../../user-guide/flow-routing/flow-accumulation/">Accumulation</a>
        <span class="specs">Calculate contributing area</span>
      </li>
    </ul>
  </div>

  <div class="module-group">
    <h4 class="module-header">03 // EXTRACTION</h4>
    <ul class="module-list">
      <li>
        <a href="../../user-guide/feature-extraction/streams/">Streams</a>
        <span class="specs">Vectorize stream network</span>
      </li>
      <li>
        <a href="../../user-guide/feature-extraction/basins/">Basins</a>
        <span class="specs">Delineate watersheds</span>
      </li>
      <li>
        <a href="../../user-guide/feature-extraction/flow-length/">Flow Length</a>
        <span class="specs">Longest Upstream Path</span>
      </li>
    </ul>
  </div>

</div>


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

<style>
  .md-content__inner > h1 { display: none; }
  .md-header__inner .md-header__title { opacity: 0; }
</style>