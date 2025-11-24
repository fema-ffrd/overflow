# Fill

## Purpose

Fill eliminates depressions by raising cell elevations to the lowest pour point elevation. The operation ensures water can flow from any cell to a domain boundary without encountering lower adjacent cells.

## When to Use

Filling raises depression interiors rather than modifying barrier cells. Use fill when:

- Depressions represent genuine closed basins
- Preserving barrier elevations is more important than depression interiors
- Breaching would require extensive or unrealistic terrain modification
- Processing pipeline requires fully conditioned surface with no depressions

Filling is typically applied after breaching to handle remaining depressions that breach cannot eliminate within search radius or cost constraints.

!!! note
    Filling does not resolve undefined flow in flat areas and will in fact create flat areas of undefined flow anywhere that it fills. Apply `flow_direction()` with `resolve_flats=True` after breaching and filling.

## Parameters

### input_path
Path to input DEM raster. GDAL-readable format. Single band. Float32 data type. Nodata cells handled according to `fill_holes` parameter.

### output_path
Path for output filled DEM. Written as GeoTIFF. Inherits projection and geotransform from input. When None, modifies input file in place.

### chunk_size
Tile dimension in pixels for tiled processing. Default 2048. Set to 0 or 1 for in-memory processing.

### working_dir
Directory for temporary files during tiled processing. Ignored for in-memory mode. When None, uses system temporary directory.

### fill_holes
Boolean flag controlling nodata region treatment. When False (default), nodata regions remain unchanged. When True, nodata cells assigned elevation of lowest surrounding valid cell, effectively filling nodata holes.

### progress_callback
Optional callback for monitoring long operations. See [ProgressCallback API](../../api/index.md#overflow.ProgressCallback).

## CLI Usage

```bash
overflow fill \
    --input_file dem.tif \
    --output_file dem_filled.tif
```

## Python API Usage

```python
import overflow

# Output to new file
overflow.fill(
    input_path="dem.tif",
    output_path="dem_filled.tif"
)

# Modify in place
overflow.fill(
    input_path="dem.tif",
    output_path=None
)
```

All cells in the output DEM will have a flow path to the domain boundary with monotonically decreasing or constant elevation. Unmodified cells retain their original elevations and no cells are lowered.

## See Also

- [Breach Operation](breach.md) - Alternative depression removal method
- [Flow Direction](../flow-routing/flow-direction.md) - Computing flow after conditioning
- [Complete Pipeline](../pipeline.md) - End-to-end workflow including fill
- [Fill Algorithm Details](../../algorithm-details/fill.md) - Implementation and theory