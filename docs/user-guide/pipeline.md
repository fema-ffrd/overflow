# Pipeline

## Purpose

The Pipeline command is a high-level orchestrator that executes a complete hydrological analysis workflow in a single command. It automates the sequence of terrain conditioning, flow routing, and feature extraction, managing intermediate file creation and dependency handling automatically.

## When to Use

Use the pipeline when you need to derive standard hydrographic features from a raw DEM without manually managing the sequence of individual operations. It is ideal for:

* **Batch Processing:** Automating analysis across many DEMs.
* **Standard Workflows:** Ensuring consistent processing steps.
* **Quick Analysis:** Getting from a raw DEM to streams and basins with minimal configuration.

## Workflow Steps

The pipeline executes the following operations in order:

1.  **Breach:** Removes pits by carving paths (optional, dependent on search radius).
2.  **Fill:** Fills remaining depressions to ensure hydrologic conditioning.
3.  **Flow Direction:** Computes D8 flow direction with flat resolution.
4.  **Flow Accumulation:** Calculates upstream drainage area.
5.  **Stream Extraction:** Vectorizes stream networks based on drainage area threshold.
6.  **Basin Delineation:** (Optional) Delineates watersheds for stream junctions.

## Parameters

### dem_file
Path to the input raw DEM raster. GDAL-readable format. Single band. Float32.

### output_dir
Directory where all output files will be written. The directory must exist. Existing files with generated names (e.g., `fdr.tif`) will be overwritten.

### chunk_size
Tile dimension in pixels for processing. Default 512. Set to 0 for in-memory processing.

### search_radius_ft
Search radius in **feet** for the pit breaching step. Default: 200. If set to 0, the breaching step is skipped, and the pipeline relies solely on filling. Internal conversion to cell count is performed based on the raster's spatial reference.

### max_cost
Maximum elevation cost for the breaching step. Default infinity. See [Breach](terrain-conditioning/breach.md) 

### da_sqmi
Drainage area threshold in **square miles** for stream extraction. Default: 1.0. Internal conversion to cell count is performed based on the raster's spatial reference. Determines the density of the resulting stream network.

### basins
Flag to enable watershed delineation. If set, basins are delineated for every junction node in the extracted stream network. Outputs a `basins.tif` raster and `basins.gpkg` vector.

### fill_holes
Flag to fill nodata holes in the DEM during the conditioning phase. See [Fill](terrain-conditioning/fill.md)

## Outputs
The pipeline generates the following files in the `output_dir`:

| Filename | Description | Data Type |
|----------|-------------|-----------|
| `dem_corrected.tif` | Hydrologically conditioned DEM (Breached & Filled) | Float32 |
| `fdr.tif` | D8 Flow Direction Raster | UInt8 |
| `accum.tif` | Flow Accumulation Raster | Int64 |
| `streams.gpkg` | GeoPackage containing `streams` (lines) and `junctions` (points) | Vector |
| `basins.tif` | Basin ID Raster (created only if `--basins` is used) | Int64 |
| `basins.gpkg` | GeoPackage containing `basins` (created only if `--basins` is used) | Vector |

## CLI Usage

Run the full pipeline with breaching, stream extraction (1 sq mi threshold), and basin delineation:

```bash
overflow pipeline \
    --dem_file raw_dem.tif \
    --output_dir ./results \
    --search_radius_ft 200 \
    --da_sqmi 1.0 \
    --basins
```

Run a fill-only pipeline (skip breaching) for a dense network (0.1 sq mi):

```bash
overflow pipeline \
    --dem_file raw_dem.tif \
    --output_dir ./results \
    --search_radius_ft 0 \
    --da_sqmi 0.1
```

## Python API Usage

The `pipeline` function is primarily a CLI convenience wrapper. To replicate the pipeline logic in Python, invoke the individual core functions sequentially. This allows for greater control over intermediate filenames and parameters.

```python
import overflow
from overflow._util.raster import sqmi_to_cell_count, feet_to_cell_count

# 1. Setup paths
dem_file = "raw_dem.tif"
output_dir = "./results"

# 2. Terrain Conditioning (Breach + Fill)
# Note: Helper functions required to convert physical units to cell counts
radius_cells = feet_to_cell_count(200, dem_file)

overflow.breach(dem_file, f"{output_dir}/dem_breached.tif", search_radius=radius_cells)
overflow.fill(f"{output_dir}/dem_breached.tif", f"{output_dir}/dem_corrected.tif")

# 3. Flow Routing
overflow.flow_direction(f"{output_dir}/dem_corrected.tif", f"{output_dir}/fdr.tif")
overflow.accumulation(f"{output_dir}/fdr.tif", f"{output_dir}/accum.tif")

# 4. Feature Extraction
# Convert 1 sq mile to cell count for threshold
threshold_cells = sqmi_to_cell_count(1.0, dem_file)

overflow.streams(
    fac_path=f"{output_dir}/accum.tif",
    fdr_path=f"{output_dir}/fdr.tif",
    output_dir=output_dir,
    threshold=threshold_cells
)

# 5. Basins (Optional)
# Uses the junctions layer from the generated streams.gpkg
overflow.basins(
    fdr_path=f"{output_dir}/fdr.tif",
    drainage_points_path=f"{output_dir}/streams.gpkg",
    output_path=f"{output_dir}/basins.tif",
    layer_name="junctions"
)
```

## See Also

  - [Breach](terrain-conditioning/breach.md) - Details on the breaching algorithm
  - [Fill](terrain-conditioning/fill.md) - Details on the filling algorithm
  - [Stream Extraction](feature-extraction/streams.md) - Details on stream vectorization
  - [Basin Delineation](feature-extraction/basins.md) - Details on watershed delineation