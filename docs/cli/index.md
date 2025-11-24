# CLI Reference

The Overflow Command Line Interface (CLI) provides access to all core hydrological analysis tools.

## Usage

All commands are invoked via the `overflow` entry point:

```bash
overflow [COMMAND] [OPTIONS]
```

-----

## Commands

### `pipeline`

**Description:** Run the complete DEM processing pipeline. This command runs a full hydrological analysis workflow: 

1.  Breach pits (optional, if search\_radius\_ft \> 0)
2.  Fill depressions
3.  Compute flow direction (with flat resolution)
4.  Calculate flow accumulation
5.  Extract streams
6.  Delineate watersheds (optional, if --basins flag is set)

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dem_file` | TEXT | - | **Yes** | Path to the GDAL supported raster dataset for the DEM.  |
| `--output_dir` | TEXT | - | **Yes** | Path to the output directory.  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 0 for in-memory processing).  |
| `--search_radius_ft` | FLOAT | 200 | No | Search radius in feet for pit breaching (0 to skip breaching).  |
| `--max_cost` | FLOAT | inf | No | Maximum cost of breach paths (total sum of elevation removed from each cell in path).  |
| `--da_sqmi` | FLOAT | 1 | No | Minimum drainage area in square miles for stream extraction.  |
| `--basins` | FLAG | False | No | Flag to enable watershed delineation.  |
| `--fill_holes` | FLAG | False | No | If set, fills holes in the DEM.  |

-----

### `breach`

**Description:** Breach pits in a DEM using least-cost paths. This command identifies pits (local minima) in the DEM and creates breach paths to allow water to flow out, minimizing the total elevation change. 

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input_file` | TEXT | - | **Yes** | Path to the GDAL supported raster dataset for the DEM.  |
| `--output_file` | TEXT | - | **Yes** | Path to the output file (must be GeoTiff).  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size.  |
| `--search_radius` | INTEGER | 200 | No | Search radius in cells.  |
| `--max_cost` | FLOAT | inf | No | Maximum cost of breach paths.  |

-----

### `fill`

**Description:** Fill depressions in a DEM using priority flood algorithm. This command fills local depressions (sinks) in the DEM to create a hydrologically conditioned surface where water can flow to the edges.

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input_file` | TEXT | - | **Yes** | Path to the GDAL supported raster dataset for the DEM.  |
| `--output_file` | TEXT | None | No | Path to the output file. If not provided, modifies input in place.  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 1 for in-memory processing).  |
| `--working_dir` | TEXT | None | No | Working directory for temporary files.  |
| `--fill_holes` | FLAG | False | No | If set, fills holes (nodata regions) in the DEM.  |

-----

### `flow-direction`

**Description:** Compute D8 flow directions from a DEM and resolve flat areas. This command calculates the steepest descent direction for each cell using the D8 algorithm, then resolves flat areas to ensure continuous flow paths. 

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input_file` | TEXT | - | **Yes** | Path to the DEM file.  |
| `--output_file` | TEXT | - | **Yes** | Path to the output file.  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 1 for in-memory processing).  |
| `--working_dir` | TEXT | None | No | Working directory for temporary files.  |
| `--no_resolve_flats` | FLAG | False | No | If set, skip resolving flat areas.  |

-----

### `accumulation`

**Description:** Calculate flow accumulation from a flow direction raster. This command computes the number of upstream cells that flow into each cell, representing drainage area in cell units.

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input_file` | TEXT | - | **Yes** | Path to the GDAL supported raster dataset for the flow direction raster.  |
| `--output_file` | TEXT | - | **Yes** | Path to the output file (must be GeoTiff).  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 1 for in-memory processing).  |

-----

### `streams`

**Description:** Extract stream networks from flow accumulation and direction rasters. This command identifies stream cells based on a flow accumulation threshold and creates vector stream lines and junction points (streams.gpkg).

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--fac_file` | TEXT | - | **Yes** | Path to the flow accumulation raster.  |
| `--fdr_file` | TEXT | - | **Yes** | Path to the flow direction raster.  |
| `--output_dir` | TEXT | - | **Yes** | Path to the output directory.  |
| `--threshold` | INTEGER | 5 | No | Minimum flow accumulation threshold (cell count) to define a stream.  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 1 for in-memory processing).  |

-----

### `basins`

**Description:** Delineate drainage basins from a flow direction raster and drainage points. This command labels each cell with the ID of its downstream drainage point, effectively delineating basin boundaries.

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--fdr_file` | TEXT | - | **Yes** | Path to the flow direction raster.  |
| `--dp_file` | TEXT | - | **Yes** | Path to the drainage points file.  |
| `--output_file` | TEXT | - | **Yes** | Path to the output file (must be GeoTiff).  |
| `--fac_file` | TEXT | None | No | Path to the flow accumulation raster (optional, for snapping).  |
| `--dp_layer` | TEXT | None | No | Name of the layer in the drainage points file.  |
| `--snap_radius_ft` | FLOAT | 0 | No | Radius in feet to snap drainage points to maximum flow accumulation.  |
| `--all_basins` | BOOL | False | No | If True, labels all basins. If False, only labels basins upstream of drainage points.  |
| `--chunk_size` | INTEGER | 2048 | No | Chunk size (use \<= 1 for in-memory processing).  |

-----

### `flow-length`

**Description:** Calculate upstream flow length (longest flow path) from drainage points. This command calculates the distance from each cell to its downstream drainage point, measured along the flow path. The output raster contains flow length values in map units.

**Options:**

| Option | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--fdr_file` | TEXT | - | **Yes** | Path to the flow direction raster.  |
| `--dp_file` | TEXT | - | **Yes** | Path to the drainage points file.  |
| `--output_raster` | TEXT | - | **Yes** | Path to the output raster file (must be GeoTiff).  |
| `--output_vector` | TEXT | None | No | Path to the output vector file (GeoPackage) for longest flow paths.  |
| `--fac_file` | TEXT | None | No | Path to the flow accumulation raster (optional, for snapping).  |
| `--dp_layer` | TEXT | None | No | Name of the layer in the drainage points file.  |
| `--snap_radius_ft` | FLOAT | 0 | No | Radius in feet to snap drainage points to maximum flow accumulation.  |
