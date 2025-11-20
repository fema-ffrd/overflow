import concurrent.futures
import math
import os
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit, types  # type: ignore[attr-defined]
from numba.typed import Dict, List  # type: ignore[attr-defined]
from numba.types import float64, int64
from osgeo import gdal, ogr
from shapely.geometry import LineString
from shapely.wkt import dumps, loads

from overflow.extract_streams.core import (
    add_downstream_junctions,
    find_node_cells,
    get_downstream_cell,
    get_stream_raster,
    nodes_to_points,
    setup_datasource,
    write_lines,
    write_points,
)
from overflow.util.constants import NEIGHBOR_OFFSETS
from overflow.util.progress import ProgressCallback, ProgressTracker, silent_callback
from overflow.util.raster import (
    RasterChunk,
    cell_to_coords,
    coords_to_cell,
    create_dataset,
    grid_hash,
    open_dataset,
    raster_chunker,
)

# Define custom types for numba
coordinate_pair_type = types.UniTuple(float64, 2)
stream_endpoint_type = types.UniTuple(types.UniTuple(float64, 2), 2)
int64_triple = types.UniTuple(types.int64, 3)
float64_quad = types.UniTuple(types.float64, 4)


# Create typed List initializers for use in JIT functions
@njit
def create_int64_list():
    return List.empty_list(int64)


@njit
def create_coordinate_pair_list():
    return List.empty_list(coordinate_pair_type)


def merge_stream_geometries(geom1, geom2, merge_type):
    """
    Merge two stream geometries using Shapely for better performance.

    Args:
        geom1: OGR Geometry of first stream
        geom2: OGR Geometry of second stream
        merge_type: Integer indicating merge type:
            0: downstream->upstream
            1: upstream->downstream
            2: upstream<-upstream
            3: downstream->downstream

    Returns:
        ogr.Geometry: Merged stream geometry
    """
    # Convert OGR geometries to Shapely
    line1 = loads(geom1.ExportToWkt())
    line2 = loads(geom2.ExportToWkt())

    # Get coordinates as arrays for faster manipulation
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)

    # Create merged coordinates based on merge type
    if merge_type == 0:  # downstream->upstream
        merged_coords = coords1 + coords2[1:]
    elif merge_type == 1:  # upstream->downstream
        merged_coords = coords2 + coords1[1:]
    elif merge_type == 2:  # upstream<-upstream
        merged_coords = list(reversed(coords2[:-1])) + coords1
    else:  # downstream->downstream
        merged_coords = list(reversed(coords1)) + coords2[1:]

    # Create merged geometry
    merged_line = LineString(merged_coords)

    # Convert back to OGR geometry
    merged_geom = ogr.CreateGeometryFromWkt(dumps(merged_line))

    return merged_geom


@njit
def get_other_endpoint(stream_info, want_upstream):
    """
    Get the other endpoint of a stream.

    Args:
        stream_info: tuple of (fid, x, y, is_upstream)
        want_upstream: boolean, whether we want the upstream point

    Returns:
        tuple: (x, y) coordinates of the requested endpoint
    """
    # If this is an upstream point and we want downstream (or vice versa),
    # we need to look up the other endpoint in our lookup structure
    x = stream_info[1]
    y = stream_info[2]
    is_upstream = stream_info[3]

    if bool(is_upstream) == bool(want_upstream):
        # This is the endpoint we want
        return (x, y)
    else:
        # This is not the endpoint we want - return None
        return (-1, -1)


@njit(nogil=True)
def process_tile(
    fac: np.ndarray,
    fdr: np.ndarray,
    cell_count_threshold: int,
    geotransform: tuple,
    tile_row: int,
    tile_col: int,
    chunk_size: int,
):
    """
    Process a single tile to extract stream networks.

    This function performs the following steps:
    1. Generate a stream raster based on the flow accumulation threshold.
    2. Identify node cells (confluences and sources) in the stream network.
    3. Convert node cells to geographic coordinates.
    4. Generate line features representing stream segments between nodes.

    Args:
        fac (np.ndarray): Flow accumulation raster for the tile.
        fdr (np.ndarray): Flow direction raster for the tile.
        cell_count_threshold (int): Minimum flow accumulation to be considered a stream.
        geotransform (tuple): Geotransform of the raster.
        tile_row (int): Row index of the current tile.
        tile_col (int): Column index of the current tile.
        chunk_size (int): Size of the tile.

    Returns:
        tuple: (streams_array, points, lines, tile_row, tile_col)
            - streams_array (np.ndarray): Boolean array indicating stream cells.
            - points (np.ndarray): Array of (x, y) coordinates for stream nodes.
            - lines (list): List of line features representing stream segments.
            - tile_row (int): Row index of the processed tile.
            - tile_col (int): Column index of the processed tile.
    """
    streams_array = get_stream_raster(fac, cell_count_threshold)
    node_cells = find_node_cells(streams_array, fdr)
    node_cell_indices = np.argwhere(node_cells)

    points = nodes_to_points(
        node_cell_indices, geotransform, tile_row, tile_col, chunk_size
    )

    lines = []
    for row, col in node_cell_indices:
        current_node = (row, col)
        line = []
        x, y = cell_to_coords(row, col, geotransform, tile_row, tile_col, chunk_size)
        line.append((x, y))
        length = 1
        while True:
            length += 1
            next_cell = get_downstream_cell(fdr, current_node[0], current_node[1])
            if next_cell[0] == -1:
                # Reached the edge of the tile, add one more point to allow merging
                fdr_value = fdr[current_node[0], current_node[1]]
                if fdr_value > 7:
                    lines.append(line)
                    break
                off_x, off_y = NEIGHBOR_OFFSETS[fdr_value]
                x, y = cell_to_coords(
                    current_node[0] + off_x,
                    current_node[1] + off_y,
                    geotransform,
                    tile_row,
                    tile_col,
                    chunk_size,
                )
                line.append((x, y))
                lines.append(line)
                break
            if node_cells[next_cell[0], next_cell[1]]:
                # Reached another node, end the current line
                x, y = cell_to_coords(
                    next_cell[0],
                    next_cell[1],
                    geotransform,
                    tile_row,
                    tile_col,
                    chunk_size,
                )
                line.append((x, y))
                lines.append(line)
                break
            x, y = cell_to_coords(
                next_cell[0], next_cell[1], geotransform, tile_row, tile_col, chunk_size
            )
            line.append((x, y))
            current_node = next_cell

    return streams_array, points, lines, tile_row, tile_col


@njit
def find_streams_to_merge(
    junction_points: np.ndarray,
    junction_fids: np.ndarray,
    stream_endpoints: np.ndarray,
    geotransform: tuple,
) -> tuple[list, list, list]:
    """
    Find which junctions to remove and which streams to merge using spatial indexing.
    All computation done in JIT-compiled code without GDAL dependencies.
    """
    # Create lookup structures
    junctions = Dict.empty(key_type=int64, value_type=int64)  # hash -> junction_fid
    endpoint_lookup = Dict.empty(
        key_type=int64, value_type=int64[:]
    )  # hash -> array of endpoint indices

    # Build junction lookup
    for i, junction_point in enumerate(junction_points):
        x, y = junction_point
        cell_i, cell_j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(cell_i, cell_j)
        junctions[hash_key] = junction_fids[i]

    # Build endpoint lookup
    # First count endpoints per cell for array sizing
    endpoint_counts = Dict.empty(key_type=int64, value_type=int64)
    for i, endpoint in enumerate(stream_endpoints):
        x, y = endpoint[1], endpoint[2]
        cell_i, cell_j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(cell_i, cell_j)
        if hash_key in endpoint_counts:
            endpoint_counts[hash_key] += 1
        else:
            endpoint_counts[hash_key] = 1

    # Allocate arrays and populate lookup
    for hash_key in endpoint_counts:
        endpoint_lookup[hash_key] = np.empty(endpoint_counts[hash_key], dtype=np.int64)
        endpoint_counts[hash_key] = 0  # Reuse as current index counter

    for i, endpoint in enumerate(stream_endpoints):
        x, y = endpoint[1], endpoint[2]
        cell_i, cell_j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(cell_i, cell_j)
        idx = endpoint_counts[hash_key]
        endpoint_lookup[hash_key][idx] = i
        endpoint_counts[hash_key] += 1

    # Initialize output lists
    junction_fids_to_remove = List.empty_list(int64)
    stream_pairs_to_merge = List.empty_list(int64_triple)
    new_endpoints = List.empty_list(float64_quad)

    # Process each junction point
    for i, junction_point in enumerate(junction_points):
        x, y = junction_point
        cell_i, cell_j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(cell_i, cell_j)

        # Skip if no endpoints at this location
        if hash_key not in endpoint_lookup:
            continue

        # Get endpoints at this junction
        junction_endpoint_indices = endpoint_lookup[hash_key]

        # If exactly 2 endpoints, process the junction
        if len(junction_endpoint_indices) == 2:
            idx1, idx2 = junction_endpoint_indices[0], junction_endpoint_indices[1]
            stream1 = stream_endpoints[idx1]
            stream2 = stream_endpoints[idx2]

            fid1, x1, y1, is_upstream1 = stream1
            fid2, x2, y2, is_upstream2 = stream2

            # Find other endpoints (rest of function remains the same)
            stream1_other = None
            stream2_other = None
            for j, stream_endpoint in enumerate(stream_endpoints):
                if stream_endpoint[0] == fid1 and j != idx1:
                    stream1_other = stream_endpoint
                elif stream_endpoint[0] == fid2 and j != idx2:
                    stream2_other = stream_endpoint
                if stream1_other is not None and stream2_other is not None:
                    break

            if stream1_other is None or stream2_other is None:
                continue

            # Determine merge type and new endpoints
            if not is_upstream1 and is_upstream2:  # downstream->upstream
                merge_type = 0
                new_ups = (stream1_other[1], stream1_other[2])
                new_downs = (stream2_other[1], stream2_other[2])
            elif is_upstream1 and not is_upstream2:  # upstream->downstream
                merge_type = 1
                new_ups = (stream2_other[1], stream2_other[2])
                new_downs = (stream1_other[1], stream1_other[2])
            elif is_upstream1 and is_upstream2:  # upstream<-upstream
                merge_type = 2
                new_ups = (stream2_other[1], stream2_other[2])
                new_downs = (stream1_other[1], stream1_other[2])
            else:  # downstream->downstream
                merge_type = 3
                new_ups = (stream1_other[1], stream1_other[2])
                new_downs = (stream2_other[1], stream2_other[2])

            junction_fids_to_remove.append(junctions[hash_key])
            stream_pairs_to_merge.append((int64(fid1), int64(fid2), int64(merge_type)))
            new_endpoints.append(
                (
                    float64(new_ups[0]),
                    float64(new_ups[1]),
                    float64(new_downs[0]),
                    float64(new_downs[1]),
                )
            )

    return junction_fids_to_remove, stream_pairs_to_merge, new_endpoints


def remove_tile_edge_junctions(
    geotransform: tuple,
    streams_dataset_path: str,
    streams_layer: str = "streams",
    junctions_layer: str = "junctions",
    progress_callback: ProgressCallback | None = None,
):
    """
    Remove junctions that have exactly 2 stream endpoints coincident with the junction
    and merge the connecting stream segments.
    """
    if progress_callback is None:
        progress_callback = silent_callback

    ds = ogr.Open(streams_dataset_path, gdal.GA_Update)
    streams_layer = ds.GetLayer(streams_layer)
    junctions_layer = ds.GetLayer(junctions_layer)

    # Collect all junction points and FIDs
    junction_points = []
    junction_fids = []
    for feature in junctions_layer:
        geom = feature.GetGeometryRef()  # type: ignore[attr-defined]
        junction_points.append((geom.GetX(), geom.GetY()))
        junction_fids.append(feature.GetFID())  # type: ignore[attr-defined]

    junction_points = np.array(junction_points, dtype=np.float64)  # type: ignore[assignment]
    junction_fids = np.array(junction_fids, dtype=np.int64)  # type: ignore[assignment]

    # Collect all stream endpoints
    stream_endpoints = []
    for feature in streams_layer:
        fid = feature.GetFID()  # type: ignore[attr-defined]
        geom = feature.GetGeometryRef()  # type: ignore[attr-defined]
        upstream = geom.GetPoint(0)
        downstream = geom.GetPoint(geom.GetPointCount() - 1)
        # Store both endpoints with flag indicating if it's upstream
        stream_endpoints.append((fid, upstream[0], upstream[1], 1))
        stream_endpoints.append((fid, downstream[0], downstream[1], 0))

    stream_endpoints = np.array(stream_endpoints, dtype=np.float64)  # type: ignore[assignment]

    # Get lists of what needs to be merged/deleted
    junction_fids_to_remove, stream_pairs_to_merge, new_endpoints = (
        find_streams_to_merge(
            junction_points, junction_fids, stream_endpoints, geotransform
        )
    )

    # Keep track of stream replacements
    stream_replacements: dict[int, int] = {}  # old_fid -> new_fid
    total_merges = len(stream_pairs_to_merge)

    for i, (
        (fid1, fid2, merge_type),
        (_up_x, _up_y, _down_x, _down_y),
    ) in enumerate(zip(stream_pairs_to_merge, new_endpoints), 1):
        # Get current FIDs accounting for previous merges
        current_fid1 = stream_replacements.get(fid1, fid1)
        current_fid2 = stream_replacements.get(fid2, fid2)

        # Get and merge the streams
        stream1 = streams_layer.GetFeature(current_fid1)  # type: ignore[attr-defined]
        stream2 = streams_layer.GetFeature(current_fid2)  # type: ignore[attr-defined]

        if stream1 is None or stream2 is None:
            # Skip this merge - warning would clutter output
            continue

        geom1 = stream1.GetGeometryRef().Clone()
        geom2 = stream2.GetGeometryRef().Clone()

        merged_geom = merge_stream_geometries(geom1, geom2, merge_type)

        # Create new feature
        new_feature = ogr.Feature(streams_layer.GetLayerDefn())  # type: ignore[attr-defined]
        new_feature.SetGeometry(merged_geom)
        streams_layer.CreateFeature(new_feature)  # type: ignore[attr-defined]
        new_fid = new_feature.GetFID()

        # Update stream replacements
        stream_replacements[fid1] = new_fid
        stream_replacements[fid2] = new_fid

        # For any streams that were previously merged into fid1 or fid2,
        # update their replacements to point to the new FID
        for old_fid, replacement_fid in list(stream_replacements.items()):
            if replacement_fid in (current_fid1, current_fid2):
                stream_replacements[old_fid] = new_fid

        # Delete original features
        streams_layer.DeleteFeature(current_fid1)  # type: ignore[attr-defined]
        streams_layer.DeleteFeature(current_fid2)  # type: ignore[attr-defined]

        # Cleanup
        stream1 = None
        stream2 = None
        new_feature = None

        # Report progress
        if i % 100 == 0 or i == total_merges:
            progress_callback(message=f"Merged {i}/{total_merges} segments")

    # Remove junctions
    for junction_fid in junction_fids_to_remove:
        junctions_layer.DeleteFeature(junction_fid)  # type: ignore[attr-defined]

    # Cleanup
    ds = None


def extract_streams_tiled(
    fac_path: str,
    fdr_path: str,
    output_dir: str,
    cell_count_threshold: int,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Extract stream networks from flow accumulation and flow direction rasters using a tiled approach.

    This function processes large rasters by dividing them into tiles, extracting stream networks
    for each tile in parallel, and then merging the results. The process includes the following steps:
    1. Open input flow accumulation and flow direction rasters.
    2. Create output datasets for streams raster and vector features.
    3. Process each tile in parallel to extract stream networks.
    4. Merge stream segments across tile boundaries.
    5. Write the final merged streams and junctions to a GeoPackage.

    Args:
        fac_path (str): Path to the flow accumulation raster.
        fdr_path (str): Path to the flow direction raster.
        output_dir (str): Directory to save output files.
        cell_count_threshold (int): Minimum flow accumulation to be considered a stream.
        chunk_size (int): Size of each tile for processing.
        progress_callback (ProgressCallback | None): Optional callback for progress updates.
            If None, the operation runs silently.

    Returns:
        None
    """
    # Setup progress tracking
    if progress_callback is None:
        progress_callback = silent_callback
    tracker = ProgressTracker(progress_callback, "Extract Streams", total_steps=3)

    # Open input datasets
    fac_ds = open_dataset(fac_path)
    fdr_ds = open_dataset(fdr_path)
    fac_band = fac_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    geotransform = fac_ds.GetGeoTransform()
    projection = fac_ds.GetProjection()

    # Create output datasets
    streams_raster_path = os.path.join(output_dir, "streams.tif")
    streams_ds = create_dataset(
        streams_raster_path,
        0,
        gdal.GDT_Byte,
        fac_ds.RasterXSize,
        fac_ds.RasterYSize,
        geotransform,
        projection,
    )
    streams_band = streams_ds.GetRasterBand(1)

    # Create temporary GeoPackage for vector outputs
    streams_dataset_path = os.path.join(output_dir, "streams.gpkg")
    output_ds, points_layer, lines_layer = setup_datasource(
        streams_dataset_path, fac_ds
    )

    # Set up parallel processing
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    def handle_tile_result(future):
        """
        Callback function to handle the result of processing each tile.

        This function writes the streams raster, points, and lines for each processed tile
        to their respective output datasets.

        Args:
            future (concurrent.futures.Future): The future object containing the result of tile processing.
        """
        with lock:
            streams_array, points, lines, tile_row, tile_col = future.result()
            # Write streams raster
            streams_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            streams_tile.from_numpy(streams_array)
            streams_tile.write(streams_band)
            # Write points
            write_points(points_layer, points)
            # Write lines
            write_lines(lines_layer, lines)
            task_queue.get()
            # Report progress as tiles complete
            chunk_counter[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = math.ceil(fac_band.YSize / chunk_size) * math.ceil(
        fac_band.XSize / chunk_size
    )

    tracker.update(1, "Extract streams")

    # Process tiles in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fac_tile in raster_chunker(fac_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)

            fdr_tile = RasterChunk(fac_tile.row, fac_tile.col, chunk_size, 0)
            fdr_tile.read(fdr_band)

            future = executor.submit(
                process_tile,
                fac_tile.data,
                fdr_tile.data,
                cell_count_threshold,
                geotransform,
                fac_tile.row,
                fac_tile.col,
                chunk_size,
            )
            future.add_done_callback(handle_tile_result)

    # Wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    # Clean up temporary datasets
    streams_band.FlushCache()
    streams_ds = None
    output_ds.FlushCache()
    output_ds = None
    fac_ds = None
    fdr_ds = None

    tracker.update(2, "Merge segments")

    remove_tile_edge_junctions(
        geotransform, streams_dataset_path, progress_callback=progress_callback
    )

    tracker.update(3, "Add junctions")

    add_downstream_junctions(
        geotransform, streams_dataset_path, progress_callback=progress_callback
    )
