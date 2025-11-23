import concurrent.futures
import heapq
import math
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import int64, njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.breach_single_cell_pits import breach_single_cell_pits_in_chunk
from overflow.util.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_RADIUS,
    EPSILON_GRADIENT,
    NEIGHBOR_OFFSETS,
    UNVISITED_INDEX,
)
from overflow.util.progress import ProgressCallback, silent_callback
from overflow.util.raster import (
    GridCellFloat32 as GridCell,
)
from overflow.util.raster import (
    RasterChunk,
    create_dataset,
    raster_chunker,
)


@njit
def process_neighbor(
    neighbor_row: int,
    neighbor_col: int,
    row_offset: int,
    col_offset: int,
    current_cell_cost: float,
    initial_elevation: float,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    heap: list[GridCell],
    cost_multiplier: float,
    current_row: int,
    current_col: int,
) -> None:
    """
    Process a neighboring cell in a Digital Elevation Model (DEM) grid for pathfinding.

    This function calculates the cost of moving to a neighboring cell and updates the
    cost and previous cell information if the calculated cost is lower than the current
    cost of the neighbor. It also enqueues the neighbor into a priority queue if the
    cost is updated.

    Parameters:
        - next_row (int): Row index of the neighboring cell.
        - next_col (int): Column index of the neighboring cell.
        - row_offset (int): Offset of the row index of the neighboring cell in the search window.
        - col_offset (int): Offset of the column index of the neighboring cell in the search window.
        - current_cell_cost (float): Current accumulated cost to reach the current cell.
        - initial_elevation (float): Initial elevation of the pit cell.
        - dem (np.ndarray): Digital Elevation Model (DEM) array.
        - dem_no_data_value (float): No data value in the DEM.
        - costs_array (np.ndarray): 2D array storing the accumulated costs of breaching
          each cell in the grid.
        - prev_rows_array (np.ndarray): 2D array storing the row indices of the previous
          cells for each cell in the grid.
        - prev_cols_array (np.ndarray): 2D array storing the column indices of the previous
          cells for each cell in the grid.
        - heap (list[GridCell]): Priority queue storing cells to be processed.
        - cost_multiplier (float): Multiplier factor for calculating cost considering diagonal movement.
        - current_row (int): Row index of the current cell.
        - current_col (int): Column index of the current cell.

    Returns:
        None
    """
    next_elevation = dem[neighbor_row, neighbor_col]
    if next_elevation == dem_no_data_value or math.isnan(next_elevation):
        next_elevation = -np.inf  # nodata cells are treated as most negative elevation
    if next_elevation != -np.inf:
        next_cost = current_cell_cost + cost_multiplier * (
            next_elevation - initial_elevation
        )
    else:
        next_cost = current_cell_cost
    # if the cost is less than the current cost of the neighbor
    if next_cost < costs_array[neighbor_row + row_offset, neighbor_col + col_offset]:
        # update the cost and previous cell of the neighbor
        costs_array[neighbor_row + row_offset, neighbor_col + col_offset] = next_cost
        prev_rows_array[neighbor_row + row_offset, neighbor_col + col_offset] = (
            current_row
        )
        prev_cols_array[neighbor_row + row_offset, neighbor_col + col_offset] = (
            current_col
        )
        # enqueue the neighbor
        heapq.heappush(heap, GridCell(neighbor_row, neighbor_col, next_cost))


@njit
def reconstruct_path(
    breach_point_row: int,
    breach_point_col: int,
    final_elevation: float,
    init_elevation: float,
    dem: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> None:
    """
    Reconstruct the least-cost path from the final breach point to the pit cell.

    This function reconstructs the least-cost path from the final breach point back to the pit cell using the
    information stored in the previous cell arrays. It applies gradient to the DEM to create the breach path.

    Parameters:
        - breach_point_row (int): Row index of the final breach point.
        - breach_point_col (int): Column index of the final breach point.
        - final_elevation (float): Elevation of the final breach point.
        - init_elevation (float): Initial elevation of the pit cell.
        - dem (np.ndarray): Digital Elevation Model (DEM) array.
        - prev_rows_array (np.ndarray): 2D array storing the row indices of the previous
          cells for each cell in the grid.
        - prev_cols_array (np.ndarray): 2D array storing the column indices of the previous
          cells for each cell in the grid.
        - row_offset (int): Offset of the row index of the final breach point in the search window.
        - col_offset (int): Offset of the column index of the final breach point in the search window.

    Returns:
        None
    """
    path = []
    row, col = breach_point_row, breach_point_col
    while UNVISITED_INDEX not in (row, col):
        path.append((row, col))
        row, col = (
            int64(prev_rows_array[row + row_offset, col + col_offset]),
            int64(prev_cols_array[row + row_offset, col + col_offset]),
        )
    # remove last cell in path since we don't want to modify the pit cell
    path.pop()
    path_length = len(path)
    for j, (path_row, path_col) in enumerate(path):
        # apply gradient to the dem to create the breach path
        if final_elevation == -np.inf:
            # we're breaching to a nodata cell, so don't modify the first cell
            if j > 0:
                # we're breaching to a nodata cell, so assume a small gradient
                dem[path_row, path_col] = min(
                    (init_elevation - (path_length - j) * EPSILON_GRADIENT),
                    dem[path_row, path_col],
                )
        else:
            # don't apply gradient to flat areas
            if dem[path_row, path_col] == init_elevation:
                continue
            dem[path_row, path_col] = min(
                (
                    final_elevation
                    + (init_elevation - final_elevation) * j / path_length
                ),
                dem[path_row, path_col],
            )


@njit
def breach_pits_in_chunk_least_cost(
    pit_row: int,
    pit_col: int,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    output_dem: np.ndarray,
    search_radius: int,
    max_cost: float = np.inf,
):
    """
    Compute least-cost paths for breach points in parallel within a given search radius.

    This function computes the least-cost paths for breach points in parallel within a specified search radius. It uses
    Dijkstra's algorithm to explore the neighborhood of each breach point until a breach point is found or the search
    window is exhausted. Once a breach point is found, it reconstructs the least-cost path from the final breach point
    back to the pit cell.

    Parameters:
        pit (np.ndarray): (row, column) representing the starting points for the least-cost
                           path computation.
        dem (np.ndarray): Digital Elevation Model (DEM) array.
        dem_no_data_value (float): No data value in the DEM.
        search_radius (int, optional): Search radius around each pit point within which the least-cost path is computed.

    Returns:
        None

    Raises:
        ValueError: If the search_radius is not a positive integer.

    Notes:
        - This function modifies the output_dem in-place to create the least-cost paths.
        - Parallel execution is utilized for processing multiple pits simultaneously.
        - The search_radius must be a positive integer. Only pits that can be solved within the search radius will be
            breached.
        - The DEM should have valid elevation values, and dem_no_data_value should be set accordingly for nodata cells.
    """
    if search_radius <= 0 or not isinstance(search_radius, int):
        raise ValueError("search_radius must be a positive integer")
    search_window_size = 2 * search_radius + 1
    # list of locals for each thread
    breach_point_found = False
    current_row = -1
    current_col = -1
    init_elevation = math.nan
    row_offset = -1
    col_offset = -1
    # initialize variables for the search
    current_row = pit_row
    current_col = pit_col
    current_cost = 0
    init_elevation = dem[current_row, current_col]
    row_offset = search_radius - current_row
    col_offset = search_radius - current_col
    costs_array[current_row + row_offset, current_col + col_offset] = 0
    heap = [GridCell(current_row, current_col, current_cost)]
    heapq.heapify(heap)
    breach_point_found = False
    while len(heap) > 0:
        # dequeue the cell with the lowest cost
        cell = heapq.heappop(heap)
        current_cost, current_row, current_col = (
            cell.value,
            cell.row,
            cell.col,
        )
        # if this cell can be breached, stop
        if (
            dem[current_row, current_col] < init_elevation
            or dem[current_row, current_col] == dem_no_data_value
            or math.isnan(dem[current_row, current_col])
        ):
            breach_point_found = True
            break
        # if the heap size is too large, stop
        if len(heap) >= search_window_size**2:
            break  # pit is unsolvable with max heap size
        # for each neighbor of the current cell
        for dr, dc in NEIGHBOR_OFFSETS:
            next_row, next_col = current_row + dr, current_col + dc
            # Calculate the cost considering diagonal movement
            multiplier = 1 if dr == 0 or dc == 0 else math.sqrt(2)
            is_in_bounds = (
                # check if the cell is inside the DEM
                0 <= next_row < dem.shape[0]
                and 0 <= next_col < dem.shape[1]
                # check if the cell is inside the search window
                and 0 <= next_row + row_offset < search_window_size
                and 0 <= next_col + col_offset < search_window_size
            )
            if is_in_bounds and current_cost < max_cost:
                process_neighbor(
                    next_row,
                    next_col,
                    row_offset,
                    col_offset,
                    current_cost,
                    init_elevation,
                    dem,
                    dem_no_data_value,
                    costs_array,
                    prev_rows_array,
                    prev_cols_array,
                    heap,
                    multiplier,
                    current_row,
                    current_col,
                )

    if breach_point_found:
        final_elevation = dem[current_row, current_col]
        final_elevation = (
            final_elevation
            if final_elevation != dem_no_data_value and not np.isnan(final_elevation)
            else -np.inf
        )
        reconstruct_path(
            current_row,
            current_col,
            final_elevation,
            init_elevation,
            output_dem,
            prev_rows_array,
            prev_cols_array,
            row_offset,
            col_offset,
        )
    # reset the costs and previous cells to their initial values
    costs_array.fill(np.inf)
    prev_rows_array.fill(UNVISITED_INDEX)
    prev_cols_array.fill(UNVISITED_INDEX)


@njit(nogil=True)
def breach_all_pits_in_chunk_least_cost(
    tile_row: int,
    tile_col: int,
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    search_radius: int,
    max_cost: float = np.inf,
):
    """
    Compute least-cost paths for all breach points within a given search radius. This function
    preallocates memory for the cost and backlink rasters and wraps breach_pits_least_cost
    to split the pits into chunks and avoid memory overflow.
    """
    # split the pits into chunks to avoid memory overflow
    output_dem = dem.copy()
    for pit_row, pit_col in pits:
        breach_pits_in_chunk_least_cost(
            pit_row,
            pit_col,
            dem,
            dem_no_data_value,
            costs_array,
            prev_rows_array,
            prev_cols_array,
            output_dem,
            search_radius,
            max_cost,
        )
    return output_dem, tile_row, tile_col


def _breach_paths_least_cost(
    input_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_cost: float = np.inf,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Main function to breach paths in a DEM using least cost algorithm.
    This function will tile the input DEM into chunks and breach the paths in
    each chunk in parallel using the least cost algorithm.

    Parameters
    ----------
    input_path : str
        Path to the input DEM file.
    output_path : str
        Path to the output breached DEM file.
    chunk_size : int, optional
        Size of chunks to process, by default DEFAULT_CHUNK_SIZE.
    search_radius : int, optional
        Search radius in cells for finding breach paths, by default DEFAULT_SEARCH_RADIUS.
    max_cost : float, optional
        Maximum cost (elevation removed) allowed for breaching, by default np.inf.
    progress_callback : ProgressCallback | None, optional
        Optional callback for progress reporting, by default None (silent).

    Returns
    -------
    None
    """
    if progress_callback is None:
        progress_callback = silent_callback

    input_ds = gdal.Open(input_path)
    projection = input_ds.GetProjection()
    geotransform = input_ds.GetGeoTransform()
    input_band = input_ds.GetRasterBand(1)
    input_nodata = input_band.GetNoDataValue()
    output_ds = create_dataset(
        output_path,
        input_nodata,
        gdal.GDT_Float32,
        input_ds.RasterXSize,
        input_ds.RasterYSize,
        geotransform,
        projection,
    )
    output_band = output_ds.GetRasterBand(1)

    # Threading setup
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    search_window_size = 2 * search_radius + 1

    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = math.ceil(input_band.YSize / chunk_size) * math.ceil(
        input_band.XSize / chunk_size
    )

    def handle_breach_tile_result(future):
        breached_dem, tile_row, tile_col = future.result()
        with lock:
            dem_tile = RasterChunk(tile_row, tile_col, chunk_size, search_radius)
            dem_tile.from_numpy(breached_dem)
            dem_tile.write(output_band)
            task_queue.get()
            # Report progress as tiles complete
            chunk_counter[0] += 1
            progress_callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    # Breach paths
    progress_callback(
        step_name="Breach paths", step_number=1, total_steps=1, progress=0.0
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        for chunk in raster_chunker(
            input_band,
            chunk_size=chunk_size,
            chunk_buffer_size=search_radius,
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            pits_raster = breach_single_cell_pits_in_chunk(chunk.data, input_nodata)
            pits_array = np.argwhere(pits_raster == 1)
            chunk_costs_array = np.full(
                (search_window_size, search_window_size),
                np.inf,
                dtype=np.float32,
            )
            chunk_prev_rows_array = np.full(
                (search_window_size, search_window_size),
                UNVISITED_INDEX,
                dtype=np.int64,
            )
            chunk_prev_cols_array = np.full(
                (search_window_size, search_window_size),
                UNVISITED_INDEX,
                dtype=np.int64,
            )
            future = executor.submit(
                breach_all_pits_in_chunk_least_cost,
                chunk.row,
                chunk.col,
                pits_array,
                chunk.data,
                input_nodata,
                chunk_costs_array,
                chunk_prev_rows_array,
                chunk_prev_cols_array,
                search_radius,
                max_cost,
            )
            future.add_done_callback(handle_breach_tile_result)

    # wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    # Flush cache and close datasets
    output_band.FlushCache()
    output_ds.FlushCache()
    output_band = None
    output_ds = None
    input_band = None
    input_ds = None
