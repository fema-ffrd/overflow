import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from numba.typed import List  # type: ignore[attr-defined]
from numba.types import int64  # type: ignore[attr-defined]
from osgeo import gdal, ogr, osr

from overflow.basins.core.basins import (
    drainage_points_from_file,
    upstream_neighbor_generator,
)
from overflow.util.constants import NEIGHBOR_OFFSETS
from overflow.util.queue import GridCellFloat32Queue as Queue
from overflow.util.raster import GridCellFloat32 as GridCell
from overflow.util.raster import create_dataset

gdal.UseExceptions()


@njit
def calculate_flow_distance_projected(
    direction: int, pixel_size_x: float, pixel_size_y: float
) -> float:
    """
    Calculate the physical distance for a flow step in projected coordinates.

    Parameters:
    - direction (int): The flow direction (0-7, where 0=E, 1=NE, 2=N, etc.)
    - pixel_size_x (float): The pixel width in map units.
    - pixel_size_y (float): The pixel height in map units (typically negative).

    Returns:
    - float: The physical distance of the flow step.
    """
    d_row, d_col = NEIGHBOR_OFFSETS[direction]
    dx = abs(d_col * pixel_size_x)
    dy = abs(d_row * pixel_size_y)
    return np.sqrt(dx * dx + dy * dy)  # type: ignore[no-any-return]


@njit
def calculate_flow_distance_geographic(
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    geotransform: tuple,
    semi_major: float,
) -> float:
    """
    Calculate the physical distance between two adjacent cells in geographic coordinates using Haversine.

    Parameters:
    - from_row, from_col: Starting cell
    - to_row, to_col: Ending cell
    - geotransform: GDAL geotransform tuple
    - semi_major: Semi-major axis of ellipsoid (meters)

    Returns:
    - float: The physical distance in meters.
    """
    # Convert cell centers to geographic coordinates
    lon1 = (
        geotransform[0]
        + (from_col + 0.5) * geotransform[1]
        + (from_row + 0.5) * geotransform[2]
    )
    lat1 = (
        geotransform[3]
        + (from_col + 0.5) * geotransform[4]
        + (from_row + 0.5) * geotransform[5]
    )
    lon2 = (
        geotransform[0]
        + (to_col + 0.5) * geotransform[1]
        + (to_row + 0.5) * geotransform[2]
    )
    lat2 = (
        geotransform[3]
        + (to_col + 0.5) * geotransform[4]
        + (to_row + 0.5) * geotransform[5]
    )

    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # Use semi-major axis as radius
    distance = semi_major * c
    return distance  # type: ignore[no-any-return]


@njit(parallel=True, nogil=True)
def calculate_upstream_flow_length(
    fdr: np.ndarray,
    drainage_points: dict,
    pixel_size_x: float,
    pixel_size_y: float,
    is_geographic: bool,
    geotransform: tuple,
    semi_major: float,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Calculate upstream flow length from drainage points with proper basin relationships.

    Strategy:
    1. Pre-claim all drainage point locations to prevent race conditions
    2. Each drainage point BFS upstream in parallel, stopping at:
       - Already claimed cells (from other basins)
       - nodata/edge
       - Basin labels = pre-claimed, then first-come-first-serve
    3. Build upstream graph from basin_labels

    Returns:
    - flow_length: Maximum distance calculated by any drainage point
    - basin_labels: Which drainage point owns each cell
    - basin_graph: Dict mapping dp_id -> list of upstream dp_ids
    - basin_max_cells: Dict mapping dp_id -> (row, col, max_distance)
    """
    rows, cols = fdr.shape
    flow_length = np.full_like(fdr, -1.0, dtype=np.float32)
    basin_labels = np.zeros_like(fdr, dtype=np.int64)

    # Convert drainage_points dict to list
    drainage_point_list = [(k[0], k[1], v) for k, v in drainage_points.items()]
    n_points = len(drainage_point_list)

    # Pre-allocate array for max cell stats (thread-safe for parallel writes)
    # Columns: 0=row, 1=col, 2=max_dist
    max_stats_arr = np.zeros((n_points, 3), dtype=np.float64)

    # Phase 1: Pre-claim all drainage point locations
    for i in range(n_points):
        global_row, global_col, dp_id = drainage_point_list[i]
        row = global_row - row_offset
        col = global_col - col_offset

        if 0 <= row < rows and 0 <= col < cols:
            basin_labels[row, col] = dp_id
            flow_length[row, col] = 0.0

    # Phase 2: BFS from each drainage point in parallel
    for i in prange(n_points):
        global_row, global_col, dp_id = drainage_point_list[i]
        row = global_row - row_offset
        col = global_col - col_offset

        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue

        # Track max cell for this basin
        max_dist = 0.0
        max_row, max_col = float(row), float(col)

        # Start BFS from drainage point
        queue = Queue([GridCell(row, col, 0.0)])

        while queue:
            cell = queue.pop()
            r = cell.row
            c = cell.col
            current_dist = cell.value

            # Find all upstream neighbors
            for n_row, n_col in upstream_neighbor_generator(fdr, r, c):
                # Stop if already claimed by a different basin
                if (
                    basin_labels[n_row, n_col] != 0
                    and basin_labels[n_row, n_col] != dp_id
                ):
                    continue

                # Calculate distance based on coordinate system
                if is_geographic:
                    dist_increment = calculate_flow_distance_geographic(
                        r, c, n_row, n_col, geotransform, semi_major
                    )
                else:
                    direction = fdr[n_row, n_col]
                    dist_increment = calculate_flow_distance_projected(
                        direction, pixel_size_x, pixel_size_y
                    )
                new_dist = current_dist + dist_increment

                # Update if unclaimed or we found a longer path
                if basin_labels[n_row, n_col] == 0:
                    # First to claim this cell
                    basin_labels[n_row, n_col] = dp_id
                    flow_length[n_row, n_col] = new_dist
                    queue.push(GridCell(n_row, n_col, new_dist))
                    # Track max
                    if new_dist > max_dist:
                        max_dist = new_dist
                        max_row, max_col = float(n_row), float(n_col)
                elif basin_labels[n_row, n_col] == dp_id:
                    # Already ours, update if longer
                    if new_dist > flow_length[n_row, n_col]:
                        flow_length[n_row, n_col] = new_dist
                        queue.push(GridCell(n_row, n_col, new_dist))
                        # Track max
                        if new_dist > max_dist:
                            max_dist = new_dist
                            max_row, max_col = float(n_row), float(n_col)

        # Store in thread-safe array
        max_stats_arr[i, 0] = max_row
        max_stats_arr[i, 1] = max_col
        max_stats_arr[i, 2] = max_dist

    # After parallel section: Convert array to typed dict
    # Create the dict and let Numba infer the tuple type
    basin_max_cells = {}
    for i in range(n_points):
        _, _, dp_id = drainage_point_list[i]
        basin_max_cells[dp_id] = (
            max_stats_arr[i, 0],
            max_stats_arr[i, 1],
            max_stats_arr[i, 2],
        )

    # Phase 3: Build upstream graph by checking basin relationships
    # First build forward graph (who flows into whom)
    flows_into = {}

    for i in range(n_points):
        global_row, global_col, dp_id = drainage_point_list[i]
        row = global_row - row_offset
        col = global_col - col_offset

        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue

        # Check if this drainage point flows into another basin
        # by following the flow direction downstream from our DP
        flow_dir = fdr[row, col]
        if flow_dir < 8:  # Valid flow direction
            d_row, d_col = NEIGHBOR_OFFSETS[flow_dir]
            next_row = row + d_row
            next_col = col + d_col

            if 0 <= next_row < rows and 0 <= next_col < cols:
                downstream_basin = basin_labels[next_row, next_col]
                # If downstream cell belongs to another basin, we flow into it
                if downstream_basin != 0 and downstream_basin != dp_id:
                    flows_into[dp_id] = downstream_basin

    # Now invert to create upstream graph: basin_graph[X] = list of basins that flow into X
    basin_graph = {}
    for i in range(n_points):
        _, _, dp_id = drainage_point_list[i]
        basin_graph[dp_id] = List.empty_list(int64)

    # Populate the inverted graph
    for upstream_basin, downstream_basin in flows_into.items():
        if downstream_basin in basin_graph:
            basin_graph[downstream_basin].append(upstream_basin)

    return flow_length, basin_labels, basin_graph, basin_max_cells


@njit
def find_all_most_upstream_basins(dp_id: int, basin_graph: dict) -> List:
    """
    Find ALL terminal upstream basins plus the current basin.

    Returns basins to check for the longest path:
    - Always includes the current basin (dp_id)
    - Includes all upstream basins that have no further upstream connections

    Returns:
    - List of basin IDs to check (current basin + all terminal upstream basins)
    """
    result = List.empty_list(int64)

    # Always include the current basin
    result.append(dp_id)

    # Find all basins upstream of this one
    all_upstream = {dp_id}
    to_visit = List.empty_list(int64)
    to_visit.append(dp_id)

    while len(to_visit) > 0:
        current = to_visit.pop()

        # Get upstream basins for this basin
        if current in basin_graph:
            upstream_list = basin_graph[current]
            for upstream_id in upstream_list:
                if upstream_id not in all_upstream:
                    all_upstream.add(upstream_id)
                    to_visit.append(upstream_id)

    # Now find all terminal basins (basins with no upstream connections)
    for basin_id in all_upstream:
        if basin_id == dp_id:
            continue  # Already added

        # Check if this basin has no upstream connections
        has_upstream = False
        if basin_id in basin_graph:
            if len(basin_graph[basin_id]) > 0:
                has_upstream = True

        # If no upstream connections, it's a terminal basin
        if not has_upstream:
            result.append(basin_id)

    return result


@njit
def trace_path_from_cell(
    fdr: np.ndarray,
    start_row: int,
    start_col: int,
    drainage_point: tuple,
) -> list:
    """
    Trace a path downstream from a starting cell to a drainage point.

    Parameters:
    - fdr: Flow direction raster
    - start_row: Starting row
    - start_col: Starting column
    - drainage_point: Target (row, col)

    Returns:
    - List of (row, col) tuples representing the path
    """
    rows, cols = fdr.shape
    path = [(start_row, start_col)]
    current_row, current_col = start_row, start_col

    # Safety limit to prevent infinite loops
    max_iterations = rows * cols
    iterations = 0

    while (current_row, current_col) != drainage_point:
        iterations += 1
        if iterations > max_iterations:
            break

        # Get the flow direction
        direction = fdr[current_row, current_col]
        if direction >= 8:  # Undefined or nodata
            break

        # Move downstream
        d_row, d_col = NEIGHBOR_OFFSETS[direction]
        next_row = current_row + d_row
        next_col = current_col + d_col

        # Check bounds
        if not (0 <= next_row < rows and 0 <= next_col < cols):
            break

        # Add to path
        path.append((next_row, next_col))
        current_row, current_col = next_row, next_col

    return path


@njit
def calculate_path_distance_projected(
    path: list, pixel_size_x: float, pixel_size_y: float
) -> float:
    """
    Calculate the total distance along a path in projected coordinates.

    Parameters:
    - path: List of (row, col) tuples
    - pixel_size_x: Pixel width
    - pixel_size_y: Pixel height

    Returns:
    - Total distance along the path
    """
    total_dist = 0.0
    for i in range(1, len(path)):
        prev_row, prev_col = path[i - 1]
        curr_row, curr_col = path[i]
        d_row = curr_row - prev_row
        d_col = curr_col - prev_col
        dx = abs(d_col * pixel_size_x)
        dy = abs(d_row * pixel_size_y)
        total_dist += np.sqrt(dx * dx + dy * dy)
    return total_dist


@njit
def calculate_path_distance_geographic(
    path: list,
    geotransform: tuple,
    semi_major: float,
) -> float:
    """
    Calculate the total distance along a path in geographic coordinates using Haversine formula.

    Parameters:
    - path: List of (row, col) tuples
    - geotransform: GDAL geotransform tuple
    - semi_major: Semi-major axis of ellipsoid (meters)

    Returns:
    - Total distance along the path in meters
    """
    total_dist = 0.0
    for i in range(1, len(path)):
        prev_row, prev_col = path[i - 1]
        curr_row, curr_col = path[i]

        # Convert to geographic coordinates (cell centers)
        lon1 = (
            geotransform[0]
            + (prev_col + 0.5) * geotransform[1]
            + (prev_row + 0.5) * geotransform[2]
        )
        lat1 = (
            geotransform[3]
            + (prev_col + 0.5) * geotransform[4]
            + (prev_row + 0.5) * geotransform[5]
        )
        lon2 = (
            geotransform[0]
            + (curr_col + 0.5) * geotransform[1]
            + (curr_row + 0.5) * geotransform[2]
        )
        lat2 = (
            geotransform[3]
            + (curr_col + 0.5) * geotransform[4]
            + (curr_row + 0.5) * geotransform[5]
        )

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        distance = semi_major * c
        total_dist += distance

    return total_dist


@njit
def trace_longest_flow_path(
    fdr: np.ndarray,
    drainage_point: tuple,
    dp_id: int,
    basin_graph: dict,
    basin_max_cells: dict,
    pixel_size_x: float,
    pixel_size_y: float,
    is_geographic: bool,
    geotransform: tuple,
    semi_major: float,
) -> list:
    """
    Trace the longest flow path from the most distant cell to a drainage point.

    Finds ALL most upstream basins, traces paths from the max cell in each,
    calculates the full path distance for each, and returns the longest path.

    Parameters:
    - fdr (np.ndarray): Flow direction raster.
    - drainage_point (tuple): The (row, col) of the drainage point.
    - dp_id (int): The ID of the drainage point.
    - basin_graph (dict): Graph of upstream basin relationships.
    - basin_max_cells (dict): Dict mapping dp_id -> (row, col, max_distance)
    - pixel_size_x: Pixel width
    - pixel_size_y: Pixel height
    - is_geographic: Whether coordinates are geographic (lat/lon)
    - geotransform: GDAL geotransform tuple
    - semi_major: Semi-major axis of ellipsoid (meters)

    Returns:
    - list: List of (row, col) tuples representing the path from farthest point to drainage point.
    """
    # Find ALL most upstream basins in this watershed
    most_upstream_basins = find_all_most_upstream_basins(dp_id, basin_graph)

    # Try all most upstream basins and find the longest path
    longest_path = [(drainage_point[0], drainage_point[1])]
    max_path_distance = 0.0

    for i in range(len(most_upstream_basins)):
        basin_id = most_upstream_basins[i]

        # Get the pre-calculated max cell for this basin
        if basin_id not in basin_max_cells:
            continue

        max_row, max_col, max_dist = basin_max_cells[basin_id]

        # Skip if no upstream cells found in this basin
        if max_dist == 0.0:
            continue

        # Trace path from this max cell to drainage point
        path = trace_path_from_cell(fdr, int(max_row), int(max_col), drainage_point)

        # Calculate the actual distance along the full traced path
        if is_geographic:
            path_distance = calculate_path_distance_geographic(
                path, geotransform, semi_major
            )
        else:
            path_distance = calculate_path_distance_projected(
                path, pixel_size_x, pixel_size_y
            )

        # Keep the longest path (by actual traced path distance)
        if path_distance > max_path_distance:
            max_path_distance = path_distance
            longest_path = path

    return longest_path


def is_geographic(srs: osr.SpatialReference) -> bool:
    """Check if a spatial reference system is geographic (lat/lon)."""
    return srs.IsGeographic() == 1  # type: ignore[no-any-return]


def calculate_geographic_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, srs: osr.SpatialReference
) -> float:
    """
    Calculate distance between two geographic coordinates using Haversine formula.
    Returns distance in meters.

    Parameters:
    - lat1, lon1: First point (degrees)
    - lat2, lon2: Second point (degrees)
    - srs: Spatial reference system to get ellipsoid parameters from
    """
    # Get ellipsoid parameters from CRS
    radius = srs.GetSemiMajor()  # semi-major axis in meters

    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    distance = radius * c
    return distance  # type: ignore[no-any-return]


def longest_flow_path_from_file(
    fdr_filepath: str,
    drainage_points_file: str,
    output_file: str,
    layer_name: str | None = None,
) -> None:
    """
    Calculate the longest flow path (upstream flow length) from drainage points.

    NOTE: This function loads the entire raster into memory. For large rasters,
    a tiled implementation would be needed (TODO).

    Parameters:
    - fdr_filepath (str): The path to the flow direction raster file.
    - drainage_points_file (str): The path to the drainage points file (OGR compatible format).
    - output_file (str): The path to save the flow length raster (GeoTIFF).
    - layer_name (str | None): The name of the layer in the drainage points file to read.
                              If None, the first layer in the file will be used. Default is None.

    Returns:
    - None

    Description:
    This function reads the flow direction raster and drainage points from files,
    calculates the upstream flow length from each drainage point independently, and saves:
    1. A GeoTIFF raster with flow lengths
    2. A GeoPackage with vector lines representing the longest flow path for each drainage point

    For projected CRS: distances are in the native units
    For geographic CRS: distances are calculated in meters using Haversine formula
    """
    # Read drainage points
    drainage_points = drainage_points_from_file(
        fdr_filepath, drainage_points_file, layer_name, True
    )

    # Get geotransform and projection
    fdr_ds = gdal.Open(fdr_filepath)
    if fdr_ds is None:
        raise ValueError("Could not open flow direction raster file")

    gt = fdr_ds.GetGeoTransform()
    projection = fdr_ds.GetProjection()
    pixel_size_x = abs(gt[1])  # Width of a pixel
    pixel_size_y = abs(gt[5])  # Height of a pixel

    # Check if CRS is geographic
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    is_geo = is_geographic(srs)

    # Load entire raster into memory
    fdr = fdr_ds.GetRasterBand(1).ReadAsArray()

    # Get ellipsoid parameters for distance calculations
    semi_major = srs.GetSemiMajor()

    # Calculate upstream flow length
    flow_length, basin_labels, basin_graph, basin_max_cells = (
        calculate_upstream_flow_length(
            fdr,
            drainage_points,
            pixel_size_x,
            pixel_size_y,
            is_geo,
            gt,
            semi_major,
        )
    )

    # Create output raster dataset
    out_ds = create_dataset(
        output_file,
        -1.0,  # nodata value for float32
        gdal.GDT_Float32,
        fdr.shape[1],
        fdr.shape[0],
        gt,
        projection,
    )
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(flow_length)
    out_band.FlushCache()
    out_ds.FlushCache()
    out_band = None
    out_ds = None

    # Create vector output for longest flow paths
    vector_output = output_file.replace(".tif", "_paths.gpkg")
    create_longest_flow_path_vectors(
        fdr,
        drainage_points,
        gt,
        projection,
        vector_output,
        is_geo,
        srs,
        basin_graph,
        basin_max_cells,
        pixel_size_x,
        pixel_size_y,
    )

    # Clean up
    fdr_ds = None


def create_longest_flow_path_vectors(
    fdr: np.ndarray,
    drainage_points: dict,
    geotransform: tuple,
    projection: str,
    output_file: str,
    is_geographic: bool,
    srs_input: osr.SpatialReference,
    basin_graph: dict,
    basin_max_cells: dict,
    pixel_size_x: float,
    pixel_size_y: float,
) -> None:
    """
    Create vector lines representing the longest flow path for each drainage point.

    Parameters:
    - fdr: Flow direction raster
    - drainage_points: Dictionary of drainage points
    - geotransform: GDAL geotransform
    - projection: WKT projection string
    - output_file: Output GeoPackage path
    - is_geographic: Whether the CRS is geographic
    - srs_input: Spatial reference system for ellipsoid parameters
    - basin_graph: Graph of upstream basin relationships
    - basin_max_cells: Dict mapping dp_id -> (row, col, max_distance)
    - pixel_size_x: Pixel width
    - pixel_size_y: Pixel height
    """
    # Create output driver
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(output_file)

    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Create layer
    layer = ds.CreateLayer("longest_flow_paths", srs, ogr.wkbLineString)

    # Add fields
    layer.CreateField(ogr.FieldDefn("dp_id", ogr.OFTInteger))

    field_length_m = ogr.FieldDefn("length_m", ogr.OFTReal)
    layer.CreateField(field_length_m)

    field_length_ft = ogr.FieldDefn("length_ft", ogr.OFTReal)
    layer.CreateField(field_length_ft)

    # Get ellipsoid parameters for geographic distance calculations
    semi_major = srs_input.GetSemiMajor()

    # Process each drainage point
    for (row, col), dp_id in drainage_points.items():
        # Trace the longest flow path
        path = trace_longest_flow_path(
            fdr,
            (row, col),
            dp_id,
            basin_graph,
            basin_max_cells,
            pixel_size_x,
            pixel_size_y,
            is_geographic,
            geotransform,
            semi_major,
        )

        if len(path) < 2:
            continue  # Skip if no upstream flow

        # Convert path to coordinates and calculate distance
        coords = []
        total_distance_meters = 0.0

        for i, (r, c) in enumerate(path):
            # Convert raster coordinates to map coordinates (cell centers)
            x = (
                geotransform[0]
                + (c + 0.5) * geotransform[1]
                + (r + 0.5) * geotransform[2]
            )
            y = (
                geotransform[3]
                + (c + 0.5) * geotransform[4]
                + (r + 0.5) * geotransform[5]
            )
            coords.append((x, y))

            # Calculate segment distance
            if i > 0:
                if is_geographic:
                    # For geographic CRS, use Haversine formula
                    dist_m = calculate_geographic_distance(
                        coords[i - 1][1],
                        coords[i - 1][0],  # lat1, lon1
                        coords[i][1],
                        coords[i][0],  # lat2, lon2
                        srs_input,
                    )
                    total_distance_meters += dist_m
                else:
                    # For projected CRS, calculate Euclidean distance
                    dx = coords[i][0] - coords[i - 1][0]
                    dy = coords[i][1] - coords[i - 1][1]
                    total_distance_meters += np.sqrt(dx * dx + dy * dy)

        # Create line geometry
        line = ogr.Geometry(ogr.wkbLineString)
        for x, y in coords:
            line.AddPoint(float(x), float(y))
        line.SetCoordinateDimension(2)

        # Create feature
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(line)
        feature.SetField("dp_id", int(dp_id))

        # Set distances (always in meters and feet)
        feature.SetField("length_m", float(total_distance_meters))
        feature.SetField("length_ft", float(total_distance_meters * 3.28084))

        layer.CreateFeature(feature)
        feature = None

    # Clean up
    ds = None
