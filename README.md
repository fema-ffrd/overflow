# Overflow

Overflow is a high-performance Python library for hydrological terrain analysis that specializes in processing massive Digital Elevation Models (DEMs) through parallel, tiled algorithms. Unlike traditional GIS tools, Overflow is built from the ground up for large-scale data processing.

## Why Overflow?

### Performance at Scale
- **Parallel Processing**: Every algorithm is designed for parallel execution using Numba
- **Memory-Efficient Tiling**: Process DEMs larger than RAM through sophisticated tiled algorithms that maintain accuracy across tile boundaries
- **Flexible Processing Modes**: Choose between in-memory processing for speed on smaller datasets or tiled processing for massive datasets

### Key Technical Advantages
- **Larger Size Limits**: Unlike existing open source hydrology tools like pysheds or proprietary ArcGIS tools, Overflow can process DEMs of excessive size with a much smaller memory footprint through its tiled algorithms
- **True Parallelism**: Most GRASS GIS tools, while memory efficient, are single-threaded. Overflow achieves true parallel processing through Numba
- **Programmable First**: Built as a proper Python library with both high-level and low-level APIs, not just a collection of command-line tools
- **Modern Algorithms**: Implements state-of-the-art approaches like:
  - Priority-flood depression filling
  - Least-cost path breaching
  - Graph-based flat resolution that maintains drainage patterns
  - Parallel flow accumulation that correctly handles tile boundaries

### When to Use Overflow

Choose Overflow when you need to:
- Process very large DEMs (10,000+ pixels in any dimension)
- Integrate hydrological processing into automated pipelines
- Leverage multiple CPU cores for parallel processing
- Handle datasets too large for traditional GIS tools
- Maintain programmatic control over the processing pipeline

### When Other Tools Might Be Better

Stick with traditional tools when:
- Working with small DEMs interactively
- Needing a GUI interface
- Requiring extensive visualization capabilities
- Processing speed isn't critical

## Example Use Cases

- **Large-Scale Hydrology**: Process high resolution, continental-scale, DEMs for flood modeling or watershed analysis
- **Automated Processing**: Integrate into data pipelines for batch processing multiple DEMs
- **High-Performance Computing**: Leverage parallel processing for time-critical applications
- **Memory-Constrained Environments**: Process massive datasets on machines with limited RAM

Overflow provides a comprehensive, scalable solution for extracting hydrological features from Digital Elevation Models. The entire pipeline, from initial DEM preprocessing through to stream network and watershed extraction, is designed to handle massive datasets while maintaining accuracy across tile boundaries. The result is a complete toolkit that takes you from raw DEM to finished hydrological products without size limitations or performance bottlenecks. 

## Key Features

### Core Tools

- **DEM Breaching**: Implements least-cost-path based breach path algorithm to eliminate depressions and create a hydrologically correct DEM.
- **DEM Depression Filling**: An implementation of (https://arxiv.org/abs/1606.06204). Fills depressions in DEMs using a parallel priority-flood algorithm while preserving natural drainage patterns.
- **Flow Direction**: Calculates D8 flow direction using parallel processing.
- **Flow Direction Flat Resolution**: An implementation of (https://www.sciencedirect.com/science/article/abs/pii/S0098300421002971). Resolves flow directions in flat areas using gradient away from higher terrain and towards lower terrain.
- **Flow Accumulation**: An implementation of (https://www.sciencedirect.com/science/article/abs/pii/S1364815216304984). Computes flow accumulation using a parallel, tiled approach that correctly handles flow across tile boundaries.
- **Stream Network Extraction**: Delineates stream networks based on flow accumulation thresholds with proper handling of stream connectivity across tiles.
- **Basin Delineation**: Performs watershed delineation using a parallel approach that maintains basin connectivity across tile boundaries.

### Tiled Processing

All algorithms in Overflow are designed to process DEMs in tiles, enabling the handling of datasets larger than available RAM. The algorithms maintain correctness across tile boundaries through sophisticated edge handling and graph-based approaches.

### Parallel Processing

Overflow utilizes parallel processing at multiple levels:
- Tile-level parallelism where multiple tiles are processed concurrently
- Within-tile parallelism using Numba for CPU acceleration

### Depression Handling

Overflow provides two approaches for handling depressions in DEMs:
1. **Breaching**: Uses a least-cost path algorithm to create drainage paths through barriers
2. **Filling**: Implements a parallel priority-flood algorithm to fill depressions while preserving natural drainage patterns

### Flow Direction in Flat Areas

The library implements an advanced flat resolution algorithm based on:
- Gradient away from higher terrain
- Gradient towards lower terrain
- Combination of both gradients to create realistic flow patterns

### Memory Efficiency

The tiled approach allows processing of very large datasets with minimal memory requirements:
- Each tile is processed independently
- Only tile edges are kept in memory for cross-tile connectivity
- Efficient data structures minimize memory overhead

## Installation

### Recommended Installation

The recommended approach is to use conda/mamba for system dependencies (GDAL, Numba) and pip for installing Overflow:

```bash
# Create a new conda environment with required system dependencies
conda create -n overflow python=3.11 gdal=3.8.4 numba=0.59.0 numpy=1.26.4 -c conda-forge

# Activate the environment
conda activate overflow

# Install overflow from PyPI
pip install overflow-hydro
```

## Requirements

**System Dependencies:**
- GDAL >= 3.8

**Python Dependencies (automatically installed via pip):**
- Python >= 3.11
- NumPy >= 1.26
- Numba >= 0.59
- Click >= 8.0
- Rich >= 13.0
- Shapely >= 2.0

## Performance Considerations

- Choose chunk sizes based on available RAM and dataset size
- Larger chunk sizes generally provide better performance but require more memory
- In some cases, the tiled approach may be slower than in-memory processing for small datasets but enables processing of much larger ones

## Output Formats

- All raster outputs are in GeoTIFF format
- Stream networks are saved as GeoPackage files containing both vector lines and junction points
- Watershed boundaries are saved as both raster (GeoTIFF) and vector (GeoPackage) formats


## Basic Usage

## Command Line Interface

Overflow provides a comprehensive command line interface for processing DEMs and performing hydrological analysis:

### Full DEM Processing Pipeline

```bash
overflow process-dem \
    --dem_file input.tif \
    --output_dir results \
    --chunk_size 2000 \
    --search_radius_ft 200 \
    --da_sqmi 1 \
    --basins \
    --fill_holes
```

### Individual Operations

#### Breach Single Cell Pits
```bash
overflow breach-single-cell-pits \
    --input_file dem.tif \
    --output_file breached.tif \
    --chunk_size 2000
```

#### Breach Paths (Least Cost)
```bash
overflow breach-paths-least-cost \
    --input_file dem.tif \
    --output_file breached.tif \
    --chunk_size 2000 \
    --search_radius 200 \
    --max_cost 100
```

#### Fill Depressions
```bash
overflow fill-depressions \
    --dem_file dem.tif \
    --output_file filled.tif \
    --chunk_size 2000 \
    --working_dir temp \
    --fill_holes
```

#### Calculate Flow Direction
```bash
overflow flow-direction \
    --input_file dem.tif \
    --output_file flowdir.tif \
    --chunk_size 2000
```

#### Fix Flats in Flow Direction
```bash
overflow fix-flats \
    --dem_file dem.tif \
    --fdr_file flowdir.tif \
    --output_file flowdir_fixed.tif \
    --chunk_size 2000 \
    --working_dir temp
```

#### Calculate Flow Accumulation
```bash
overflow flow-accumulation \
    --fdr_file flowdir.tif \
    --output_file flowacc.tif \
    --chunk_size 2000
```

#### Extract Stream Network
```bash
overflow extract-streams \
    --fac_file flowacc.tif \
    --fdr_file flowdir.tif \
    --output_dir streams \
    --cell_count_threshold 5 \
    --chunk_size 2000
```

#### Delineate Watersheds
```bash
overflow label-watersheds \
    --fdr_file flowdir.tif \
    --dp_file points.gpkg \
    --output_file basins.tif \
    --chunk_size 2000 \
    --all_basins
```

### Key Parameters

- `chunk_size`: Controls tile size for processing. Larger values use more memory but may be faster. Default is 2000.
- `search_radius`: Distance to search for breach paths (in cells).
- `search_radius_ft`: Distance to search for breach paths (in feet, automatically converted to cells).
- `da_sqmi`: Minimum drainage area in square miles for stream extraction.
- `cell_count_threshold`: Minimum number of cells draining to a point to be considered a stream.
- `max_cost`: Maximum elevation that can be removed when breaching paths.
- `working_dir`: Directory for temporary files during processing.
- `fill_holes`: Flag to fill no-data holes in the DEM.
- `all_basins`: Flag to delineate all watersheds, not just those upstream of drainage points.

### Notes

- All operations support both in-memory (chunk_size ≤ 0) and tiled processing modes
- For large datasets, use tiled processing with an appropriate chunk_size
- Most operations output GeoTIFF format except streams/watersheds which also output GeoPackage vector files

## Python API

### Individual Operations

#### DEM Pit Processing

```python
from overflow import breach_single_cell_pits, breach_paths_least_cost

# Breach single cell pits
breach_single_cell_pits(
    input_path="dem.tif",
    output_path="breached_pits.tif",
    chunk_size=2000
)

# Breach paths using least cost algorithm
breach_paths_least_cost(
    input_path="dem.tif",
    output_path="breached_paths.tif",
    chunk_size=2000,
    search_radius=200,  # cells to search for breach path
    max_cost=100       # maximum elevation that can be removed
)
```

#### Depression Filling

```python
from overflow import fill_depressions, fill_depressions_tiled

# In-memory depression filling
fill_depressions(
    dem_file="dem.tif",
    output_file="filled.tif",
    fill_holes=True    # fill no-data holes in DEM
)

# Tiled depression filling for large DEMs
fill_depressions_tiled(
    dem_file="dem.tif",
    output_file="filled.tif",
    chunk_size=2000,
    working_dir="temp",
    fill_holes=True
)
```

#### Flow Direction and Flat Resolution

```python
from overflow import flow_direction
from overflow.fix_flats.core import fix_flats_from_file
from overflow.fix_flats.tiled import fix_flats_tiled

# Calculate flow direction
flow_direction(
    input_path="dem.tif",
    output_path="flowdir.tif",
    chunk_size=2000
)

# Fix flats in-memory
fix_flats_from_file(
    dem_file="dem.tif",
    fdr_file="flowdir.tif",
    output_file="flowdir_fixed.tif"
)

# Fix flats using tiled approach
fix_flats_tiled(
    dem_file="dem.tif",
    fdr_file="flowdir.tif",
    output_file="flowdir_fixed.tif",
    chunk_size=2000,
    working_dir="temp"
)
```

#### Flow Accumulation

```python
from overflow import flow_accumulation, flow_accumulation_tiled

# Calculate flow accumulation in-memory
flow_accumulation(
    fdr_path="flowdir.tif",
    output_path="flowacc.tif"
)

# Calculate flow accumulation using tiled approach
flow_accumulation_tiled(
    fdr_file="flowdir.tif",
    output_file="flowacc.tif",
    chunk_size=2000
)
```

#### Stream Network Extraction

```python
from overflow import extract_streams, extract_streams_tiled

# Extract streams in-memory
extract_streams(
    fac_path="flowacc.tif",
    fdr_path="flowdir.tif",
    output_dir="results",
    cell_count_threshold=1000  # minimum drainage area in cells
)

# Extract streams using tiled approach
extract_streams_tiled(
    fac_file="flowacc.tif",
    fdr_file="flowdir.tif",
    output_dir="results",
    cell_count_threshold=1000,
    chunk_size=2000
)
```

#### Watershed Delineation

```python
from overflow.basins.core import label_watersheds_from_file, drainage_points_from_file
from overflow.basins.tiled import label_watersheds_tiled

# Get drainage points from vector file
drainage_points = drainage_points_from_file(
    fdr_filepath="flowdir.tif",
    drainage_points_file="points.gpkg",
    layer_name=None    # use first layer if None
)

# Delineate watersheds in-memory
label_watersheds_from_file(
    fdr_filepath="flowdir.tif",
    drainage_points_file="points.gpkg",
    output_file="basins.tif",
    all_basins=True,   # delineate all basins vs only those upstream of points
    dp_layer=None
)

# Delineate watersheds using tiled approach
label_watersheds_tiled(
    fdr_filepath="flowdir.tif",
    drainage_points=drainage_points,
    output_file="basins.tif",
    chunk_size=2000,
    all_basins=True
)
```

### Processing Modes

- All major operations support both in-memory and tiled processing
- Tiled mode: Use functions with `_tiled` suffix and specify `chunk_size > 0`

### Memory Considerations

- In-memory mode loads entire dataset into RAM
- Tiled mode processes data in chunks, using less memory
- Larger chunk sizes generally improve performance but require more memory
- Working directory required for temporary files in tiled mode

### Output Formats

- Most operations output GeoTIFF rasters
- Stream network extraction produces:
  - `streams.tif`: Raster representation of streams
  - `streams.gpkg`: Vector representation with streams and junction points
- Watershed delineation produces:
  - `basins.tif`: Raster representation of watersheds
  - `basins.gpkg`: Vector representation of watershed boundaries

### Unit Conversion Utilities

```python
from overflow.util.raster import sqmi_to_cell_count, feet_to_cell_count

# Convert feet to cell count based on DEM resolution
cells = feet_to_cell_count(200, "dem.tif")  # 200ft to cells

# Convert square miles to cell count
cells = sqmi_to_cell_count(1, "dem.tif")    # 1 sq mile to cells
```
