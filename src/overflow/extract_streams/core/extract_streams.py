import os
import sys
from rich.console import Console
from osgeo import gdal, ogr, osr
import numpy as np
from numba import njit, prange
from overflow.basins.core import upstream_neighbor_generator
from overflow.util.constants import NEIGHBOR_OFFSETS
from overflow.util.raster import (
    create_dataset,
    open_dataset,
    cell_to_coords,
    coords_to_cell,
    grid_hash,
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
    for i in prange(streams_array.shape[0]):  # pylint: disable=not-an-iterable
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
    for i in prange(fac.shape[0]):  # pylint: disable=not-an-iterable
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
            "junctions", srs=srs, geom_type=ogr.wkbPoint
        )
        lines_layer = output_ds.CreateLayer(
            "streams", srs=srs, geom_type=ogr.wkbLineString
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
    for i in prange(node_cell_indices.shape[0]):  # pylint: disable=not-an-iterable
        row, col = node_cell_indices[i]
        x, y = cell_to_coords(row, col, geotransform, tile_row, tile_col, chunk_size)
        points[i][0] = x
        points[i][1] = y
    return points


def add_downstream_junctions(
    geotransform: tuple,
    streams_dataset_path: str,
    streams_layer: str = "streams",
    junctions_layer: str = "junctions",
):
    """
    Add junctions one cell upstream from the downstream end of any stream
    that does not already have a junction at its downstream end.
    """
    console = Console()
    is_a_tty = sys.stdout.isatty()

    ds = ogr.Open(streams_dataset_path, gdal.GA_Update)
    streams_layer = ds.GetLayer(streams_layer)
    junctions_layer = ds.GetLayer(junctions_layer)

    # Get all existing junction locations
    junction_hashes = set()
    for feature in junctions_layer:
        geom = feature.GetGeometryRef()
        x, y = geom.GetX(), geom.GetY()
        i, j = coords_to_cell(x, y, geotransform)
        hash_key = grid_hash(i, j)
        junction_hashes.add(hash_key)

    # Find streams without downstream junctions
    total_streams = streams_layer.GetFeatureCount()
    junctions_to_add = []

    with console.status(
        "[bold green]Finding streams without downstream junctions..."
    ) as status:
        for i, feature in enumerate(streams_layer, 1):
            if is_a_tty:
                status.update(f"[bold green]Checking streams: {i}/{total_streams}")
            else:
                print(f"Checking streams: {i}/{total_streams}", end="\r", flush=True)

            geom = feature.GetGeometryRef()
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

    if not is_a_tty:
        print()

    # Add new junctions
    total_to_add = len(junctions_to_add)
    with console.status("[bold green]Adding downstream junctions...") as status:
        for i, point in enumerate(junctions_to_add, 1):
            if is_a_tty:
                status.update(f"[bold green]Adding junctions: {i}/{total_to_add}")
            else:
                print(f"Adding junctions: {i}/{total_to_add}", end="\r", flush=True)

            # Create new junction
            new_junction = ogr.Feature(junctions_layer.GetLayerDefn())
            point_geom = ogr.Geometry(ogr.wkbPoint)
            point_geom.AddPoint(point[0], point[1])
            new_junction.SetGeometry(point_geom)
            junctions_layer.CreateFeature(new_junction)
            new_junction = None

    if not is_a_tty:
        print()

    # Cleanup
    ds = None


def draw_lines(
    fdr: np.ndarray,
    node_cells: np.ndarray,
    node_cell_indices: np.ndarray,
    geotransform: tuple,
):
    """
    Generate line features representing stream segments between node cells.

    Args:
        fdr (np.ndarray): Flow direction raster.
        node_cells (np.ndarray): Boolean array indicating node cells.
        node_cell_indices (np.ndarray): Array of (row, col) indices of node cells.
        geotransform (tuple): Geotransform of the raster.

    Returns:
        list: List of line features, where each line is a list of (x, y) coordinate tuples.
    """
    lines = []
    console = Console()
    is_a_tty = sys.stdout.isatty()
    with console.status("[bold green]Processing Streams: ") as status:
        for i in range(node_cell_indices.shape[0]):
            if is_a_tty:
                status.update(
                    f"[bold green]Processing Streams: {i + 1}/{node_cell_indices.shape[0]}"
                )
            else:
                print(
                    f"Processing Streams: {i + 1}/{node_cell_indices.shape[0]}",
                    end="\r",
                    flush=True,
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
                    line.append(
                        cell_to_coords(next_cell[0], next_cell[1], geotransform)
                    )
                    lines.append(line)
                    break
                line.append(cell_to_coords(next_cell[0], next_cell[1], geotransform))
                current_node = next_cell
        return lines


def extract_streams(
    fac_path: str, fdr_path: str, output_dir: str, cell_count_threshold: int
):
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
    """
    print(f"Extracting streams with threshold {cell_count_threshold}")

    # Open input datasets
    fac_ds = open_dataset(fac_path)
    fdr_ds = open_dataset(fdr_path)
    fac_band = fac_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    fac = fac_band.ReadAsArray()
    fdr = fdr_band.ReadAsArray()
    geotransform = fac_ds.GetGeoTransform()
    projection = fac_ds.GetProjection()

    print("Get Stream Raster")
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

    print("Find Node Cells")
    node_cells = find_node_cells(streams_array, fdr)
    num_node_cells = np.sum(node_cells)
    node_cell_indices = np.argwhere(node_cells)
    print(f"Found {num_node_cells} node cells")

    points = nodes_to_points(node_cell_indices, geotransform)

    print("Creating GeoPackage")
    streams_dataset_path = os.path.join(output_dir, "streams.gpkg")
    output_ds, points_layer, lines_layer = setup_datasource(
        streams_dataset_path, fac_ds
    )

    print("Writing Points")
    write_points(points_layer, points)

    print("Drawing Lines")
    lines = draw_lines(fdr, node_cells, node_cell_indices, geotransform)

    print("Writing Lines")
    write_lines(lines_layer, lines)

    # Clean up
    output_ds = None
    fac_ds = None
    fdr_ds = None

    print("Adding Downstream Junctions")
    add_downstream_junctions(geotransform, streams_dataset_path)

    print("Done")
