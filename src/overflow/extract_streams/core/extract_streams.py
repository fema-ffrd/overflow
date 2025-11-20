import os

import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from osgeo import gdal, ogr, osr

from overflow.basins.core import upstream_neighbor_generator
from overflow.util.constants import NEIGHBOR_OFFSETS
from overflow.util.progress import ProgressCallback, silent_callback
from overflow.util.raster import (
    cell_to_coords,
    coords_to_cell,
    create_dataset,
    grid_hash,
    open_dataset,
)


@njit
def get_downstream_cell(fdr, i, j):
    """
    Get the downstream cell coordinates based on the flow direction.

    Args:
        fdr (np.ndarray): Flow direction raster.
        i (int): Current cell row index.
        j (int): Current cell column index.

    Returns:
        tuple: (row, col) of the downstream cell, or (-1, -1) if out of bounds or invalid flow direction.
    """
    fdr_value = fdr[i, j]
    if fdr_value > 7:  # Invalid flow direction
        return -1, -1
    off_x, off_y = NEIGHBOR_OFFSETS[fdr_value]
    # Check if the downstream cell is within bounds
    if (
        i + off_x < 0
        or i + off_x >= fdr.shape[0]
        or j + off_y < 0
        or j + off_y >= fdr.shape[1]
    ):
        return -1, -1

    return i + off_x, j + off_y


@njit(parallel=True)
def find_node_cells(streams_array, fdr):
    """
    Identify node cells in the stream network.

    Node cells are defined as:
    1. Cells with more than one upstream neighbor (confluence)
    2. Cells with no upstream neighbors (source)

    Args:
        streams_array (np.ndarray): Boolean array indicating stream cells.
        fdr (np.ndarray): Flow direction raster.

    Returns:
        np.ndarray: Boolean array indicating node cells.
    """
    node_cells = np.zeros_like(streams_array, dtype=np.bool_)
    # Iterate over all cells in parallel
    for i in prange(streams_array.shape[0]):
        for j in range(streams_array.shape[1]):
            if streams_array[i, j]:
                upstream_count = 0
                for neighbor in upstream_neighbor_generator(fdr, i, j):
                    if streams_array[neighbor[0], neighbor[1]]:
                        upstream_count += 1
                # Mark as node if it's a confluence or source
                if upstream_count > 1 or (upstream_count == 0 and streams_array[i, j]):
                    node_cells[i, j] = True
    return node_cells


@njit(parallel=True)
def get_stream_raster(fac: np.ndarray, cell_count_threshold: int):
    """
    Create a boolean raster indicating stream cells based on flow accumulation.

    Args:
        fac (np.ndarray): Flow accumulation raster.
        cell_count_threshold (int): Minimum flow accumulation to be considered a stream.

    Returns:
        np.ndarray: Boolean array where True indicates a stream cell.
    """
    streams_array = np.empty_like(fac, dtype=np.bool_)
    for i in prange(fac.shape[0]):
        for j in range(fac.shape[1]):
            streams_array[i, j] = fac[i, j] >= cell_count_threshold
    return streams_array


def setup_datasource(path: str, fac_ds: gdal.Dataset):
    """
    Set up a GeoPackage data source for storing stream features.

    Args:
        path (str): Path to create the GeoPackage.
        fac_ds (gdal.Dataset): Flow accumulation dataset to copy projection from.

    Returns:
        tuple: (output_ds, points_layer, lines_layer) - The created data source and layers.

    Raises:
        ValueError: If there's an error reading the spatial reference.
    """
    output_ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    srs = osr.SpatialReference()
    try:
        srs.ImportFromWkt(fac_ds.GetProjection())
        points_layer = output_ds.CreateLayer(
            "junctions", srs=srs, geom_type=ogr.wkbPoint25D
        )
        lines_layer = output_ds.CreateLayer(
            "streams", srs=srs, geom_type=ogr.wkbLineString25D
        )
        return output_ds, points_layer, lines_layer
    except RuntimeError as e:
        output_ds = None
        raise ValueError(
            "Error reading spatial reference from flow accumulation raster"
        ) from e


def write_points(points_layer, points):
    """
    Write point features to the specified layer.

    Args:
        points_layer (ogr.Layer): The layer to write points to.
        points (list): List of (x, y) coordinate tuples.
    """
    for x, y in points:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x, y)
        feature = ogr.Feature(points_layer.GetLayerDefn())
        feature.SetGeometry(point)
        points_layer.CreateFeature(feature)
        feature.Destroy()


def write_lines(lines_layer, lines):
    """
    Write line features to the specified layer.

    Args:
        lines_layer (ogr.Layer): The layer to write lines to.
        lines (list): List of lists, where each inner list contains (x, y) coordinate tuples.
    """
    for line in lines:
        line_geom = ogr.Geometry(ogr.wkbLineString)
        for x, y in line:
            line_geom.AddPoint(x, y)
        feature = ogr.Feature(lines_layer.GetLayerDefn())
        feature.SetGeometry(line_geom)
        lines_layer.CreateFeature(feature)
        feature.Destroy()


@njit(parallel=True)
def nodes_to_points(
    node_cell_indices: np.ndarray,
    geotransform: tuple,
    tile_row=0,
    tile_col=0,
    chunk_size=0,
):
    """
    Convert node cell indices to geographic coordinates.

    Args:
        node_cell_indices (np.ndarray): Array of (row, col) indices of node cells.
        geotransform (tuple): Geotransform of the raster.
        tile_row (int, optional): Row index of the current tile. Defaults to 0.
        tile_col (int, optional): Column index of the current tile. Defaults to 0.
        chunk_size (int, optional): Size of each tile. Defaults to 0.

    Returns:
        np.ndarray: Array of (x, y) coordinates for each node cell.
    """
    points = np.empty_like(node_cell_indices, dtype=np.float64)
    for i in prange(node_cell_indices.shape[0]):
        row, col = node_cell_indices[i]
        x, y = cell_to_coords(row, col, geotransform, tile_row, tile_col, chunk_size)
        points[i][0] = x
        points[i][1] = y
    return points


def add_downstream_junctions(
    geotransform: tuple,
    streams_dataset_path: str,
    streams_layer_name: str = "streams",
    junctions_layer: str = "junctions",
    progress_callback: ProgressCallback | None = None,
):
    """
    Add junctions one cell upstream from the downstream end of any stream
    that does not already have a junction at its downstream end.
    """
    if progress_callback is None:
        progress_callback = silent_callback

    ds = ogr.Open(streams_dataset_path, gdal.GA_Update)
    streams_layer = ds.GetLayer(streams_layer_name)
    junctions_layer = ds.GetLayer(junctions_layer)

    # Get all existing junction locations
    junction_hashes = set()
    for feature in junctions_layer:
        geom = feature.GetGeometryRef()  # type: ignore[attr-defined]
        x, y = geom.GetX(), geom.GetY()
        i, j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(i, j)
        junction_hashes.add(hash_key)

    # Find streams without downstream junctions
    junctions_to_add = []
    total_streams = streams_layer.GetFeatureCount()

    for idx, feature in enumerate(streams_layer, 1):
        geom = feature.GetGeometryRef()  # type: ignore[attr-defined]
        point_count = geom.GetPointCount()

        if point_count < 2:
            continue

        # Get the downstream endpoint
        endpoint = geom.GetPoint(point_count - 1)
        i, j = coords_to_cell(endpoint[0], endpoint[1], geotransform)
        downstream_hash = grid_hash(i, j)

        # Get second to last point (one cell upstream from endpoint)
        junction_point = geom.GetPoint(point_count - 2)
        i, j = coords_to_cell(junction_point[0], junction_point[1], geotransform)
        hash_key = grid_hash(i, j)

        if downstream_hash not in junction_hashes:
            junctions_to_add.append(junction_point)
            junction_hashes.add(
                hash_key
            )  # Add to set to avoid duplicates at same location

        # Report progress
        if idx % 100 == 0 or idx == total_streams:
            progress_callback(message=f"Processed {idx}/{total_streams} streams")

    # Add new junctions
    for point in junctions_to_add:
        # Create new junction
        new_junction = ogr.Feature(junctions_layer.GetLayerDefn())  # type: ignore[attr-defined]
        point_geom = ogr.Geometry(ogr.wkbPoint)
        point_geom.AddPoint(point[0], point[1])
        new_junction.SetGeometry(point_geom)
        junctions_layer.CreateFeature(new_junction)  # type: ignore[attr-defined]
        new_junction = None

    # Cleanup
    ds = None


def draw_lines(
    fdr: np.ndarray,
    node_cells: np.ndarray,
    node_cell_indices: np.ndarray,
    geotransform: tuple,
    progress_callback: ProgressCallback | None = None,
):
    """
    Generate line features representing stream segments between node cells.

    Args:
        fdr (np.ndarray): Flow direction raster.
        node_cells (np.ndarray): Boolean array indicating node cells.
        node_cell_indices (np.ndarray): Array of (row, col) indices of node cells.
        geotransform (tuple): Geotransform of the raster.
        progress_callback (ProgressCallback | None): Optional callback for progress updates.

    Returns:
        list: List of line features, where each line is a list of (x, y) coordinate tuples.
    """
    if progress_callback is None:
        progress_callback = silent_callback

    lines = []
    total_nodes = node_cell_indices.shape[0]
    for i in range(total_nodes):
        # Report progress
        progress_callback(
            step_name="Process streams",
            step_number=i + 1,
            total_steps=total_nodes,
            progress=(i + 1) / total_nodes,
            message=f"Stream {i + 1}/{total_nodes}",
        )
        row, col = node_cell_indices[i]
        current_node = (row, col)
        line = []
        x, y = cell_to_coords(row, col, geotransform)
        line.append((x, y))
        length = 1
        while True:
            length += 1
            next_cell = get_downstream_cell(fdr, current_node[0], current_node[1])
            if next_cell[0] == -1:  # Reached edge of raster
                lines.append(line)
                break
            if node_cells[next_cell[0], next_cell[1]]:  # Reached another node
                line.append(cell_to_coords(next_cell[0], next_cell[1], geotransform))
                lines.append(line)
                break
            line.append(cell_to_coords(next_cell[0], next_cell[1], geotransform))
            current_node = next_cell
    return lines


def extract_streams(
    fac_path: str,
    fdr_path: str,
    output_dir: str,
    cell_count_threshold: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Extract stream networks from flow accumulation and flow direction rasters.

    This function performs the following steps:
    1. Read flow accumulation and flow direction rasters.
    2. Generate a stream raster based on the flow accumulation threshold.
    3. Identify node cells (confluences and sources) in the stream network.
    4. Convert node cells to geographic coordinates.
    5. Generate line features representing stream segments between nodes.
    6. Write the resulting points (nodes) and lines (streams) to a GeoPackage.

    Args:
        fac_path (str): Path to the flow accumulation raster.
        fdr_path (str): Path to the flow direction raster.
        output_dir (str): Directory to save output files.
        cell_count_threshold (int): Minimum flow accumulation to be considered a stream.

    Returns:
        None
    """
    if progress_callback is None:
        progress_callback = silent_callback

    # Open input datasets
    fac_ds = open_dataset(fac_path)
    fdr_ds = open_dataset(fdr_path)
    fac_band = fac_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    fac = fac_band.ReadAsArray()
    fdr = fdr_band.ReadAsArray()
    geotransform = fac_ds.GetGeoTransform()
    projection = fac_ds.GetProjection()

    progress_callback(
        step_name="Get stream raster",
        step_number=1,
        total_steps=6,
        progress=0.17,
        message=f"Extract streams with threshold {cell_count_threshold}",
    )
    streams_array = get_stream_raster(fac, cell_count_threshold)

    # Save stream raster
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
    streams_band.WriteArray(streams_array)

    progress_callback(
        step_name="Find node cells",
        step_number=2,
        total_steps=6,
        progress=0.33,
        message="Identifying stream confluences and sources",
    )
    node_cells = find_node_cells(streams_array, fdr)
    num_node_cells = np.sum(node_cells)
    node_cell_indices = np.argwhere(node_cells)

    points = nodes_to_points(node_cell_indices, geotransform)

    progress_callback(
        step_name="Create GeoPackage",
        step_number=3,
        total_steps=6,
        progress=0.5,
        message=f"Found {num_node_cells} node cells",
    )
    streams_dataset_path = os.path.join(output_dir, "streams.gpkg")
    output_ds, points_layer, lines_layer = setup_datasource(
        streams_dataset_path, fac_ds
    )

    progress_callback(
        step_name="Write points",
        step_number=4,
        total_steps=6,
        progress=0.67,
        message="Writing node points",
    )
    write_points(points_layer, points)

    progress_callback(
        step_name="Draw lines",
        step_number=5,
        total_steps=6,
        progress=0.83,
        message="Tracing stream lines",
    )
    lines = draw_lines(
        fdr, node_cells, node_cell_indices, geotransform, progress_callback
    )

    write_lines(lines_layer, lines)

    # Clean up
    del output_ds
    fac_ds = None
    fdr_ds = None

    progress_callback(
        step_name="Add downstream junctions",
        step_number=6,
        total_steps=6,
        progress=1.0,
        message="Adding junctions",
    )
    add_downstream_junctions(
        geotransform, streams_dataset_path, progress_callback=progress_callback
    )
