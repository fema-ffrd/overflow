# Basin Delineation

## Purpose

Basin delineation assigns each cell a label identifying its downstream drainage point. The output raster contains unique basin IDs corresponding to its most immediate downstream drainage point. Drainage points can be snapped to the flow accumulation raster and the provided drainage points are attributed in place with the generated basin ID and their downstream basin ID.

## Parameters

### fdr_path
Path to flow direction raster. GDAL-readable format. Single band. UInt8 data type. Values 0-7 represent valid flow directions. Nodata value 9. Must have all non-nodata cells with defined flow directions.

### drainage_points_path
Path to drainage points vector file. Any GDAL-readable vector format (Shapefile, GeoPackage, GeoJSON). Must contain point geometries. Points located outside raster extent or on nodata cells are invalid and ignored.

### output_path
Path for basin raster output. Written as GeoTIFF. Int64 data type. Inherits projection and geotransform from flow direction input. Cell values have generated basin IDs. Nodata value -1.

### chunk_size
Tile dimension in pixels. Default 2048. Set to 0 or 1 for in-memory processing.

### all_basins
Boolean flag controlling basin labeling scope. When False (default), only cells draining to specified drainage points are labeled. Cells draining elsewhere receive nodata (-1). When True, all cells labeled including those not draining to specified points (e.g., off map flow).

### fac_path
Path to flow accumulation raster for drainage point snapping. When provided with nonzero `snap_radius`, drainage points are moved to cell with maximum accumulation within snap radius. Ensures points snap to stream channels. When None, no snapping performed.

### snap_radius
Snapping search radius in cells. When positive and `fac_path` provided, drainage points moved to maximum accumulation cell within radius. When 0, no snapping. Default 0.

### layer_name
Name of layer in vector file to read for the drainage points. When None, uses first layer. Relevant for multi-layer formats like GeoPackage.

### progress_callback
Optional callback for monitoring long operations. See [ProgressCallback API](../../api/index.md#overflow.ProgressCallback).

## CLI Usage

```bash
overflow basins \
    --fdr fdr.tif \
    --drainage_points streams.gpkg \
    --output basins.tif \
    --layer_name junctions
```

## Python API Usage

```python
import overflow

overflow.basins(
    fdr_path="fdr.tif",
    drainage_points_path="streams.gpkg",
    output_path="basins.tif"
    layer_name="junctions"
)
```

The output cell values contain generated basin ID corresponding to their downstream drainage point. All cells draining to the same point share the same label. Nodata cells (-1) are propagated from the input. When `all_basins=True` basins draining off map but not to a provided draininge point are also included. The provided drainage points will be attributed with their basin IDs and downstream basin IDs.

## Drainage Point Snapping

Snapping corrects drainage point locations that fall slightly off stream channels due to digitization error or CRS mismatch.

Snapping requires accumulation raster with values representing upstream cell count. Without snapping, drainage points used at exact provided coordinates—may not correspond to stream channel locations.

## Drainage Point Requirements

Drainage points must satisfy:

- Point geometry type
- Located within raster extent
- Located on valid (non-nodata) cells

## Using Stream Junctions as Drainage Points

Common workflow combines stream extraction with basin delineation:

```python
import overflow

# Extract streams with junctions
overflow.streams("accum.tif", "fdr.tif", "./output", threshold=100)

# Delineate basins at junctions
overflow.basins(
    fdr_path="fdr.tif",
    drainage_points_path="./output/streams.gpkg",
    output_path="basins.tif",
    layer_name="junctions"
)
```

## See Also

- [Flow Direction](../flow-routing/flow-direction.md) - Computing flow directions for basin delineation
- [Stream Extraction](streams.md) - Generating drainage points at stream junctions
- [Longest Flow Path](flow-length.md) - Computing flow path lengths within basins
- [Complete Pipeline](../pipeline.md) - End-to-end workflow
- [Basin Delineation Algorithm](../../algorithm-details/basin-delineation.md) - Implementation details