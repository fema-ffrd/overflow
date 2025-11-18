import concurrent.futures
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from osgeo import gdal, ogr
from shapely.geometry import Polygon

from overflow.util.constants import NEIGHBOR_OFFSETS_4, NEIGHBOR_OFFSETS_8
from overflow.util.progress import ProgressCallback, silent_callback
from overflow.util.raster import cell_to_coords, raster_chunker


@njit
def get_next_boundary_cell_polygons(
    boundary_cells: set, row: int, col: int, neighbor_index: int
) -> tuple:
    """
    Get the next boundary cell in the Square tracing algorithm for polygons.
    https://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/square.html

    Args:
    - boundary_cells (set): A set of all boundary cells in the basin.
    - row (int): The current row index.
    - col (int): The current column index.
    - neighbor_index (int): The index of the current neighbor.

    Returns:
    - tuple: The neighbor index, row and column indices of the next boundary cell.
    """
    next_row = row
    next_col = col
    # stopping condition: we must reach a new cell that is a boundary cell
    while (next_row, next_col) not in boundary_cells or (next_row, next_col) == (
        row,
        col,
    ):
        if (next_row, next_col) in boundary_cells:
            # if this is a boundary cell, go right
            neighbor_index = (neighbor_index - 1) % 4
        else:
            # if this is not a boundary cell, go left
            neighbor_index = (neighbor_index + 1) % 4
        next_row = row + NEIGHBOR_OFFSETS_4[neighbor_index][0]
        next_col = col + NEIGHBOR_OFFSETS_4[neighbor_index][1]
    return neighbor_index, next_row, next_col


@njit
def trace_boundary_polygons(
    boundary_cells_set: set,
    start_row: int,
    start_col: int,
    start_neighbor: int,
) -> list:
    """
    Trace the boundary of a polygon using the Moore-Neighbor tracing algorithm.

    Args:
    - boundary_cells_set (set): A set of all boundary cells in the polygon.
    - start_row (int): The starting row index.
    - start_col (int): The starting column index.
    - start_neighbor (int): The starting neighbor index.

    Returns:
    - list: A list of all boundary cells in the polygon ordered in counter-clockwise direction.
    """
    boundary_cells = []
    # initialize the first boundary cell with the next boundary cell
    # this ensures we have a start neighbor that will repeat
    # and give a stopping condition that will be met
    start_neighbor, start_row, start_col = get_next_boundary_cell_polygons(
        boundary_cells_set, start_row, start_col, start_neighbor
    )
    boundary_cells.append((start_row, start_col))
    neighbor_index, row, col = get_next_boundary_cell_polygons(
        boundary_cells_set, start_row, start_col, start_neighbor
    )
    # we must stop when we reach the starting cell in the same way we started
    # stopping when we reach the starting cell does not trace the entire boundary in some cases
    while (row, col) != (start_row, start_col) or neighbor_index != start_neighbor:
        boundary_cells.append((row, col))
        neighbor_index, row, col = get_next_boundary_cell_polygons(
            boundary_cells_set, row, col, neighbor_index
        )
    return boundary_cells


@njit  # type: ignore[misc]
def find_boundary_cells_watershed(
    watersheds: np.ndarray, ids: set, row_offset: int, col_offset: int
) -> tuple[list[set[tuple[int, int]]], dict[int, int]]:
    """
    Find all boundary cells for a given watershed value.

    Args:
    - watersheds (np.ndarray): The labeled watersheds array.
    - ids (set): The set of watershed values to find boundary cells for.

    Returns:
    - dict: A dictionary mapping watershed values to their boundary cells in the watershed.
    """
    # since numba does not support dictionaries of sets, we need to roll our own using a list of sets and a dictionary
    # mapping watershed values to their index in the list
    cells = []
    id_to_index = {}
    # initialize the list of sets
    for i, basin_id in enumerate(ids):
        empty_set = set()
        empty_set.add((-1, -1))
        empty_set.clear()
        cells.append(empty_set)
        id_to_index[basin_id] = i
    # a boundary cell is any cell with the given value for which at least one
    # neighbor has a different value
    for row in range(1, watersheds.shape[0] - 1):
        for col in range(1, watersheds.shape[1] - 1):
            basin_id = watersheds[row, col]
            if basin_id in ids:
                for offset in NEIGHBOR_OFFSETS_8:
                    neighbor_value = watersheds[row + offset[0], col + offset[1]]
                    if neighbor_value != basin_id:
                        cells[id_to_index[basin_id]].add(
                            (row + row_offset, col + col_offset)
                        )
                        break
    return cells, id_to_index


@njit
def find_boundary_start_cell(boundary_cells: set) -> tuple:
    """
    Find the starting cell for the boundary tracing algorithm.

    Args:
    - boundary_cells (set): A set of tuples representing boundary cell coordinates.

    Returns:
    - tuple: The starting neighbor index, row and column indices.
    """
    if not boundary_cells:
        return (-1, -1, -1)  # no boundary cells found

    start_row = np.iinfo(np.int64).max
    start_col = np.iinfo(np.int64).max

    for row, col in boundary_cells:
        if row < start_row or (row == start_row and col < start_col):
            start_row = row
            start_col = col

    return (4, start_row, start_col)  # starting neighbor is to the west


def compute_targeted_upstream_basins(graph, target_basin_ids):
    """
    Compute upstream basins for a specified set of basin IDs.

    Args:
    - graph (dict): A dictionary representing the graph where keys are basin IDs
                    and values are the downstream basin IDs.
    - target_basin_ids (set): A set of basin IDs for which to compute upstream basins.

    Returns:
    - dict: A dictionary where keys are the specified basin IDs and values are sets of all upstream
            basin IDs (including the basin itself).
    """
    # Step 1: Create a reversed graph
    reversed_graph = {}
    for upstream, downstream in graph.items():
        if downstream not in reversed_graph:
            reversed_graph[downstream] = set()
        reversed_graph[downstream].add(upstream)
        # Ensure every node is in the reversed graph
        if upstream not in reversed_graph:
            reversed_graph[upstream] = set()

    # Step 2: Compute upstream basins for each target node
    upstream_basins = {}
    for node in target_basin_ids:
        if node not in upstream_basins:
            upstream_basins[node] = compute_upstream(node, reversed_graph)

    return upstream_basins


def compute_upstream(node, reversed_graph):
    """
    Compute upstream basins for a node.

    Args:
    - node: The current node.
    - reversed_graph: The reversed graph.

    Returns:
    - set: A set of all upstream basin IDs (including the node itself).
    """
    upstream = set()
    to_visit = [node]
    while to_visit:
        current = to_visit.pop()
        if current not in upstream:
            upstream.add(current)
            to_visit.extend(reversed_graph.get(current, []))
    return upstream


@njit(nogil=True)
def process_chunk(watersheds: np.ndarray, row_offset: int, col_offset: int):
    basin_ids = set(np.unique(watersheds))
    if 0 in basin_ids:
        basin_ids.remove(0)  # nodata value
    boundary_cells_list, id_to_index = find_boundary_cells_watershed(
        watersheds, basin_ids, row_offset, col_offset
    )
    return boundary_cells_list, id_to_index


@njit(parallel=True)
def boundary_cells_to_coords(boundary_cells, gt):
    boundary_cells_coords = np.empty((len(boundary_cells), 2), dtype=np.float64)
    for index in prange(len(boundary_cells)):
        row, col = boundary_cells[index]
        boundary_cells_coords[index] = cell_to_coords(row, col, gt)
    return boundary_cells_coords


@njit
def augment_boundary_cells(boundary_cells):
    # in order to trace cell edges and not cell centers,
    # we create a new set of boundary cells that includes 8 cells along the edges of each cell
    # at half the resolution
    augmented_cells = set()
    for row, col in boundary_cells:
        augmented_cells.add((2 * row, 2 * col))
        augmented_cells.add((2 * row, 2 * col + 1))
        augmented_cells.add((2 * row, 2 * col + 2))
        augmented_cells.add((2 * row + 1, 2 * col))
        augmented_cells.add((2 * row + 2, 2 * col))
        augmented_cells.add((2 * row + 2, 2 * col + 1))
        augmented_cells.add((2 * row + 2, 2 * col + 2))
        augmented_cells.add((2 * row + 1, 2 * col + 2))
    return augmented_cells


@njit
def augment_geotransform(gt):
    # augment the geotransform to match the augmented boundary cells
    new_gt = list(gt)
    # move the top left corner up and to the left by a quarter a cell
    new_gt[0] -= gt[1] / 4
    new_gt[3] -= gt[5] / 4
    # scale the cell size by half
    new_gt[1] /= 2
    new_gt[5] /= 2
    return list(new_gt)


@njit(nogil=True)
def process_basin(basin_id, upstream_basin_boundary_cells, gt):
    augmented_boundary_cells = augment_boundary_cells(upstream_basin_boundary_cells)
    start_neighbor, start_row, start_col = find_boundary_start_cell(
        augmented_boundary_cells
    )
    augmented_boundary_cells = trace_boundary_polygons(
        augmented_boundary_cells, start_row, start_col, start_neighbor
    )
    # note, since we augmented boundary cells to the corners of each cell,
    # we need to augment the geotransform to match
    gt = augment_geotransform(gt)
    # note: augmented_boundary_cells does not repeat the first cell at the end,
    # but this is required for the polygon to be closed so we add it here
    augmented_boundary_cells.append(augmented_boundary_cells[0])
    upstream_basin_boundary_coords = boundary_cells_to_coords(
        augmented_boundary_cells, gt
    )
    return basin_id, upstream_basin_boundary_coords


def create_basin_polygons(
    watersheds_band: gdal.Band,
    graph: dict,
    chunk_size: int,
    output_filepath: str,
    gt: tuple,
    projection: str,
    progress_callback: ProgressCallback | None = None,
):
    """
    Create polygons for each basin in the watershed raster. The polygons are created by tracing the boundary of each
    basin using the Moore-Neighbor tracing algorithm. The function then creates a polygon from the boundary cells and
    saves it to a GeoPackage file.

    Args:
    - watersheds_band (gdal.Band): The watershed raster band.
    - graph (dict): The graph representing the watershed connections.
    - chunk_size (int): The size of each chunk.
    - output_filepath (str): The path to save the GeoPackage file.
    - gt (tuple): The geotransform of the raster.
    - projection (str): The projection of the raster.

    Returns:
    - None
    """
    # Use silent callback if none provided
    if progress_callback is None:
        progress_callback = silent_callback

    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()
    boundary_cells: dict[int, set[tuple[int, int]]] = {}

    def handle_chunk_result(future):
        with lock:
            boundary_cells_list, id_to_index = future.result()
            for basin_id, index in id_to_index.items():
                if basin_id in boundary_cells:
                    boundary_cells[basin_id].update(boundary_cells_list[index])
                else:
                    boundary_cells[basin_id] = boundary_cells_list[index]
            task_queue.get()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk in raster_chunker(
            watersheds_band, chunk_size, 1, progress_callback=progress_callback
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(1)

            row_offset = chunk.row * chunk_size - 1
            col_offset = chunk.col * chunk_size - 1

            future = executor.submit(process_chunk, chunk.data, row_offset, col_offset)
            future.add_done_callback(handle_chunk_result)

    # Wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    # Create the output datasource and layer for the polygons
    driver = ogr.GetDriverByName("GPKG")
    data_source = driver.CreateDataSource(output_filepath)
    srs = ogr.osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = data_source.CreateLayer("basins", srs, ogr.wkbPolygon)
    id_field = ogr.FieldDefn("BasinID", ogr.OFTInteger64)
    layer.CreateField(id_field)

    def handle_basin_result(future):
        with lock:
            try:
                basin_id, upstream_basin_boundary_coords = future.result()
                shapely_polygon = Polygon(upstream_basin_boundary_coords)
                wkb_geom = shapely_polygon.wkb
                ogr_geometry = ogr.CreateGeometryFromWkb(wkb_geom)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("BasinID", basin_id)
                feature.SetGeometry(ogr_geometry)
                layer.CreateFeature(feature)
                feature = None
            except Exception as e:
                print("Warning: Failed to create polygon for basin", e)
            finally:
                task_queue.get()

    upstream_basins_dict = compute_targeted_upstream_basins(
        graph, set(boundary_cells.keys())
    )
    index = 0
    num_basins = len(boundary_cells)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for basin_id in boundary_cells:
            index += 1
            # Report progress
            progress_callback(
                phase="Processing Basins",
                step=index,
                total_steps=num_basins,
                progress=index / num_basins,
                message=f"Basin {index}/{num_basins}",
            )
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(1)

            upstream_basins = upstream_basins_dict[basin_id]
            # union the boundary cells of all upstream basins
            upstream_basin_boundary_cells = set.union(
                *[
                    boundary_cells[upstream_basin]
                    for upstream_basin in upstream_basins
                    if upstream_basin in boundary_cells
                ]
            )

            future = executor.submit(
                process_basin, basin_id, upstream_basin_boundary_cells, gt
            )
            future.add_done_callback(handle_basin_result)

    # Wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    data_source = None
