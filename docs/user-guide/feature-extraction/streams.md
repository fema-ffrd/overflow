# Stream Extraction

## Purpose

Stream extraction converts accumulation-defined stream cells into vector line features. Cells exceeding accumulation threshold are classified as streams, vectorized into polylines, and written with junction points to GeoPackage output.

## Parameters

### fac_path
Path to flow accumulation raster. GDAL-readable format. Single band. Int64 data type. Values represent upstream cell count. Nodata values (-1) excluded from stream classification.

### fdr_path
Path to flow direction raster. GDAL-readable format. Single band. UInt8 data type. Values 0-7 represent valid flow directions. Required for tracing stream connectivity and determining junction locations.

### output_dir
Directory for output files. Must exist, the function does not create the directory. Writes `streams.gpkg` containing two layers: `streams` (LineString geometries) and `junctions` (Point geometries). Overwrites existing file.

### threshold
Minimum accumulation value for stream classification. Cells with $\text{accumulation} \geq \text{threshold}$ become stream cells. Controls stream network density. Lower values extract tributaries, higher values extract main channels only.

### chunk_size
Tile dimension in pixels. Default 512. Set to 0 or 1 for in-memory processing.

### progress_callback
Optional callback for monitoring long operations. See [ProgressCallback API](../../api/index.md#overflow.ProgressCallback).

## CLI Usage

```bash
overflow streams \
    --fac accum.tif \
    --fdr fdr.tif \
    --output streams_output \
    --threshold 100
```

## Python API Usage

```python
import overflow

overflow.streams(
    fac_path="accum.tif",
    fdr_path="fdr.tif",
    output_dir="./output",
    threshold=100
)
```

The output `streams.gpkg` will contain two layers:

* **streams**: The vectorized stream network of all cells with $\text{accumulation} \geq \text{threshold}$. The verticies of the polylines will be at the grid cell centers.
* **junctions**: Points at the downstream end of each reach and the upstream most points of the stream network.

## See Also

- [Flow Accumulation](../flow-routing/flow-accumulation.md) - Computing accumulation for stream extraction
- [Basin Delineation](basins.md) - Delineating watersheds using stream junctions
- [Complete Pipeline](../pipeline.md) - End-to-end workflow
- [Stream Extraction Algorithm](../../algorithm-details/stream-extraction.md) - Implementation details