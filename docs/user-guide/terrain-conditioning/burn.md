# Burn

## Purpose

Burn imposes known elevations on a DEM inside regions taken from a second raster. The mask raster is typically binary (one class of interest) or classified (several), and each contiguous region within it is rewritten to a constant elevation, to the DEM lowered by a fixed amount, or to a statistic of the DEM beneath that region.

## When to Use

Burn applies elevation information the DEM itself does not carry. Use burn when:

- Waterbodies from a land cover or hydrography raster should be flat rather than carrying survey noise
- Bridge decks and culvert footprints obstruct flow and need to be dropped to the channel elevation beneath them
- A levee breach, excavation or fill footprint has a known design elevation
- A region's elevation should be defined relative to what is already there, such as lowering a channel by a fixed depth

Burn is a conditioning step and does not by itself remove every depression. Run [breach](breach.md) and [fill](fill.md) afterwards to produce a fully conditioned surface.

!!! note
    Burn only touches cells inside the selected regions. Every other cell, including DEM nodata cells, is copied through unchanged.

## Regions

A region is a maximal connected set of cells that share the **same** mask value and whose value is one of the selected values. Two selected classes that touch stay in separate regions, so a lake class and a wetland class that share a border are never merged.

Regions are recognized as whole even when they straddle processing tiles: a lake that spans four tiles gets one minimum, not four. See [Burn Algorithm Details](../../algorithm-details/burn.md) for how that is done.

## Parameters

### dem_path
Path to the input DEM raster. Must be GDAL-readable, single band, and must declare a nodata value.

### mask_path
Path to a binary or classified mask raster, co-registered with the DEM: same shape and same geotransform. Any GDAL-readable data type. A nodata value is recommended but not required.

### output_path
Path for the output burned DEM, written as GeoTIFF. Data type, nodata value, projection and geotransform are inherited from the input DEM.

### method
How to choose the elevation to burn.

| Method | Cell value written |
|---|---|
| `constant` | the burn value configured for the cell's mask value |
| `relative` | `dem - burn_value`, applied per cell so relief inside the region is preserved |
| `statistic` | the region's own statistic of the DEM beneath it, less `burn_offset` |

### mask_values
Which mask values identify regions, given as a comma separated string such as `"1,3"` or as a sequence. If omitted, the keys of `burn_values` are used. If neither names specific values, every non-zero, non-nodata mask value is selected, and each distinct value still forms its own regions.

### burn_values
Required by `constant` and `relative`, ignored by `statistic`. Either a mapping of mask value to burn value, written as `"1:225.5,3:210.0"` or given as a dict, or a single number applied to every selected mask value.

### statistic
Which statistic the `statistic` method computes per region: `min`, `max` or `mean`. Default `min`.

### burn_offset
Subtracted from each computed statistic, so `statistic="min"` with `burn_offset=1.0` writes each region's minimum elevation less one. Used only by the `statistic` method. Default `0.0`.

### connectivity
`8` (default) treats diagonally touching cells as one region. `4` requires a shared edge, so a diagonal chain of cells becomes several regions.

### chunk_size
Tile dimension in pixels for processing. Default 2048. Set to 1 or less for in-memory processing when the DEM fits in available RAM.

### working_dir
Directory for temporary files during tiled processing with the `statistic` method. If omitted, a system temp directory is created and removed afterwards. The `constant` and `relative` methods write no temporary files.

### progress_callback
Optional callback for monitoring long operations. See [ProgressCallback API](../../api/index.md#overflow.ProgressCallback).

## CLI Usage

Flatten every waterbody to its own minimum elevation, one foot lower, so water drains through:

```bash
overflow burn \
    --dem_file dem.tif \
    --mask_file waterbodies.tif \
    --output_file dem_burned.tif \
    --method statistic \
    --mask_values 1 \
    --statistic min \
    --burn_offset 1.0
```

Set two classes to known design elevations:

```bash
overflow burn \
    --dem_file dem.tif \
    --mask_file structures.tif \
    --output_file dem_burned.tif \
    --method constant \
    --burn_values "1:225.5,3:210.0"
```

Lower a channel class by 2 feet while preserving its shape:

```bash
overflow burn \
    --dem_file dem.tif \
    --mask_file channels.tif \
    --output_file dem_burned.tif \
    --method relative \
    --burn_values "1:2.0"
```

## Python API Usage

```python
import overflow

overflow.burn(
    dem_path="dem.tif",
    mask_path="waterbodies.tif",
    output_path="dem_burned.tif",
    method="statistic",
    mask_values="1",
    statistic="min",
    burn_offset=1.0,
)
```

`burn_values` also accepts a dict, which is often easier to build programmatically:

```python
overflow.burn(
    dem_path="dem.tif",
    mask_path="structures.tif",
    output_path="dem_burned.tif",
    method="constant",
    burn_values={1: 225.5, 3: 210.0},
)
```

## Nodata Handling

- DEM nodata cells are never written, so nodata holes survive the burn intact.
- DEM nodata cells are excluded from region statistics, so a lake with a data gap in the middle still gets the minimum of its valid cells.
- A region lying entirely over DEM nodata has no statistic and is left untouched.
- Mask nodata cells are never selected. Naming the mask's own nodata value in `mask_values` is an error.

## Performance Considerations

The `constant` and `relative` methods are per-cell functions of the DEM and mask values, so they run in a single tiled pass with no temporary raster. They are roughly as fast as copying the DEM.

The `statistic` method needs three passes and writes a temporary Int64 label raster the size of the input, so budget disk space in `working_dir` accordingly. Memory during the solve is proportional to the number of region fragments, not to raster size, so masks covering a small fraction of the DEM stay cheap even on very large rasters.

!!! warning "Integer DEMs truncate fractional burn values"
    The output carries the input DEM's data type. Burning a `mean` statistic, a `relative` amount or an offset into an integer DEM truncates the result. Convert the DEM to a floating point type first if that matters.

## See Also

- [Breach](breach.md) - Carving flow paths through elevation barriers
- [Fill](fill.md) - Depression removal using priority flood
- [Flow Direction](../flow-routing/flow-direction.md) - Computing flow routing after conditioning
- [Burn Algorithm Details](../../algorithm-details/burn.md) - Implementation and theory
