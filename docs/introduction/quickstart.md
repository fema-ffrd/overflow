# Quick Start

The pipeline command executes terrain conditioning, flow routing, and feature extraction from a DEM. Stream network output is generated automatically. Basin delineation is optional.

## Installation

Installation procedures are documented in [Installation Guide](../getting-started/installation.md).

## Pipeline Command

### CLI Usage

Basic execution:

```bash
overflow pipeline \
    --dem_file dem.tif \
    --output_dir ./results
```

Output files in `./results/`:

- `dem_breached.tif` - Breached DEM (float32)
- `dem_filled.tif` - Filled DEM (float32)
- `fdr.tif` - D8 flow directions (byte)
- `accum.tif` - Upstream cell count (int64)
- `streams.gpkg` - Vector stream network with junctions

!!! note
    The output directory must exist. It will not be created if it does not.

!!! note
    If files with the same names existing in the output directory they will be overwritten.

By default both breaching and filling is performed. The default breach search radius is 200 feet. Modify with `--search_radius_ft`:

```bash
overflow pipeline \
    --dem_file dem.tif \
    --output_dir ./results \
    --search_radius_ft 300
```

Disable breaching with `--search_radius_ft 0`. Pipeline will only fill depressions. Filling cannot be disabled.

Control stream density with `--da_sqmi` parameter (drainage area threshold in square miles):

```bash
overflow pipeline \
    --dem_file dem.tif \
    --output_dir ./results \
    --da_sqmi 0.5
```

Lower values produce denser networks. Higher values extract major channels only.

Delineate basins with the `--basins` parameter:

```bash
overflow pipeline \
    --dem_file dem.tif \
    --output_dir ./results \
    --da_sqmi 0.5 \
    --basins
```

Basins will automatically be generated at the downstream end of each reach and at the upstream most points in the stream network.

### Docker Usage

Pull image:

```bash
docker pull ghcr.io/fema-ffrd/overflow:latest
```

Execute with volume mount:

```bash
docker run -v $(pwd)/data:/mnt/data \
    ghcr.io/fema-ffrd/overflow:latest \
    pipeline \
    --dem_file /mnt/data/dem.tif \
    --output_dir /mnt/data/results
```

File paths reference container mount point `/mnt/data`.

## Python API

Sequential function calls provide fine-grained control:

```python
import overflow

overflow.breach("dem.tif", "dem_breached.tif")
overflow.fill("dem_breached.tif", "dem_filled.tif")
overflow.flow_direction("dem_filled.tif", "fdr.tif")
overflow.accumulation("fdr.tif", "fac.tif")
overflow.streams("fac.tif", "fdr.tif", "./output", threshold=100)
overflow.basins("fdr.tif", "./output/streams.gpkg", "./output", layer_name="junctions")
```

!!! note
    In the Python API, all units are in cell counts or map units. No projection or conversions are performed.

!!! note
    The streams.gpkg will contain a "junctions" layer with points at the downstream ends of each reach and at the upstream ends of the stream network.

## Input Requirements

DEM constraints:

- GDAL-readable format (e.g., GeoTIFF)
- Single-band float32

!!! warning
    A projected coordinate system with consistent horizontal and vertical units is recommended for best results.

## Documentation Structure

- [User Guide](../user-guide/terrain-conditioning/breach.md) - Operation details
- [API Reference](../api/index.md) - Function signatures
- [CLI Reference](../cli/index.md) - Command documentation
- [Algorithm Details](../algorithm-details/index.md) - Implementation and theory