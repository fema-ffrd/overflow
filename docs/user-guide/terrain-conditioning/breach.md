# Breach

## Purpose

Breach removes depressions from a DEM by carving flow paths through elevation barriers. The operation identifies local minima and creates least-cost paths to adjacent lower terrain or domain boundaries.

## When to Use

Breaching preserves natural terrain features by lowering barrier cells rather than raising depression interiors. Use breach when:

- Depressions result from data artifacts (bridges, culverts, missing data interpolation)
- Maintaining original elevation values in depression interiors is important
- Terrain contains linear barriers (roads, levees, embankments) obstructing flow

Use fill instead when depressions represent genuine closed basins or when breach paths would require unrealistic terrain modification.

!!! note
    Both breach and fill can be used together and it is generally recommended to do so.

!!! note
    Breaching does not resolve undefined flow in flat areas. Apply `flow_direction()` with `resolve_flats=True` after breaching and filling.

## Parameters

### input_path
Path to input DEM raster. Must be GDAL-readable format. Single band. Float32 data type. Nodata cells propagate through output.

### output_path
Path for output breached DEM. Written as GeoTIFF. Inherits projection, geotransform, and nodata value from input.

### chunk_size
Tile dimension in pixels for processing. Default 2048. Set to 0 for in-memory processing when DEM fits in available RAM.

### search_radius
Maximum distance in cells to search for breach targets. Larger values increase computation time but allow breaching larger obstructions.

### max_cost
Maximum elevation sum removed along breach path. Default infinity (no constraint). When finite, prevents breaching depressions requiring excessive terrain modification. Cost computed as sum of elevation decrease at each modified cell.

### progress_callback
Optional callback for monitoring long operations. See [ProgressCallback API](../../api/index.md#overflow.ProgressCallback).

## CLI Usage

```bash
overflow breach \
    --input_file dem.tif \
    --output_file dem_breached.tif \
    --search_radius 50 \
    --max_cost 10.0
```

## Python API Usage

```python
import overflow

overflow.breach(
    input_path="dem.tif",
    output_path="dem_breached.tif",
    search_radius=50,
    max_cost=10.0
)
```

The output DEM will have all depressions within `search_radius` cells of lower terrain breached. Unmodified cells retain their original elevations and no cells are raised. Nodata cells remain unchanged and are considered to be lower than any other cell.

## Performance Considerations

Large search radii substantially increase processing time. Profile with typical data to determine acceptable trade-off between breach capability and runtime.

!!! warning "Breaching May Leave Residual Pits"
    The breach algorithm may leave some pits unresolved, either because they are beyond the search radius or due to tile boundary effects where pits near tile edges may be incompletely breached. **You must run the `fill` process after breaching** to ensure that no pits remain and produce a fully hydrologically conditioned DEM.

    Within each tile, pits are processed sequentially in a deterministic order, and later breaches can benefit from earlier breach paths through the use of the min() operation. However, tiles are processed in parallel, and pits located at tile boundaries may be processed differently by adjacent tiles, leading to potential artifacts at tile edges. These artifacts are resolved by the subsequent fill operation.

## See Also

- [Fill](fill.md) - Depression removal method using priority flood fill algorithm
- [Flow Direction](../flow-routing/flow-direction.md) - Computing flow routing after conditioning
- [Pipeline](../pipeline.md) - End-to-end workflow including breach
- [Breach Algorithm Details](../../algorithm-details/breach.md) - Implementation and theory