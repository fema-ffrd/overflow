import concurrent.futures
import math
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.flow_accumulation.core.flow_accumulation import (
    get_next_cell,
    single_tile_flow_accumulation,
)
from overflow.util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
)
from overflow.util.perimeter import get_tile_perimeter
from overflow.util.progress import ProgressCallback, ProgressTracker, silent_callback
from overflow.util.raster import (
    RasterChunk,
    create_dataset,
    open_dataset,
    raster_chunker,
)

from .global_state import GlobalState


@njit(nogil=True)
def calculate_flow_accumulation_tile(
    flow_direction: np.ndarray, tile_row: int, tile_col: int
):
    """
    Calculate flow accumulation for a single tile.

    This function processes a single tile of the flow direction raster to compute
    flow accumulation and extract perimeter information needed for global processing.

    Args:
        flow_direction (np.ndarray): Flow direction data for the tile.
        tile_row (int): Row index of the tile in the grid.
        tile_col (int): Column index of the tile in the grid.

    Returns:
        tuple: Contains flow accumulation data and perimeter information.
    """
    # Compute flow accumulation and links for the tile
    flow_accumulation, links = single_tile_flow_accumulation(flow_direction)

    # Extract perimeter information for global processing
    flow_acc_perimeter = get_tile_perimeter(flow_accumulation)
    flow_dir_perimeter = get_tile_perimeter(flow_direction)
    links_row_perimeter = get_tile_perimeter(links[:, :, 0])
    links_col_perimeter = get_tile_perimeter(links[:, :, 1])

    return (
        flow_accumulation,
        flow_acc_perimeter,
        flow_dir_perimeter,
        links_row_perimeter,
        links_col_perimeter,
        tile_row,
        tile_col,
    )


@njit(nogil=True)
def finalize_flow_accumulation(
    flow_acc: np.ndarray,
    flow_dir: np.ndarray,
    global_acc: dict,
    global_offset: dict,
    tile_row: int,
    tile_col: int,
    global_state: GlobalState,
):
    """
    Finalize flow accumulation for a tile using global accumulation data.

    This function adjusts the flow accumulation of a tile based on the global
    accumulation data, propagating additional flow through the tile.

    Args:
        flow_acc (np.ndarray): Initial flow accumulation for the tile.
        flow_dir (np.ndarray): Flow direction data for the tile.
        global_acc (dict): Global accumulation values.
        global_offset (dict): Global offset values.
        tile_row (int): Row index of the tile.
        tile_col (int): Column index of the tile.
        global_state (GlobalState): Global state object containing grid information.

    Returns:
        tuple: Updated flow accumulation array and tile indices.
    """
    rows, cols = flow_acc.shape
    chunk_size = global_state.chunk_size

    # Process each entry in the global accumulation and offset
    for global_index in global_offset:
        # Convert global index to local tile coordinates
        global_row = global_index // (global_state.num_cols * chunk_size)
        global_col = global_index % (global_state.num_cols * chunk_size)
        local_row = global_row - tile_row * chunk_size
        local_col = global_col - tile_col * chunk_size

        # Check if the cell is within this tile
        if 0 <= local_row < rows and 0 <= local_col < cols:
            # Propagate the additional accumulation downstream
            current_row, current_col = local_row, local_col
            current_dir = flow_dir[current_row, current_col]
            while current_dir != FLOW_DIRECTION_NODATA:
                flow_acc[current_row, current_col] += global_offset[global_index]
                current_row, current_col, current_dir = get_next_cell(
                    flow_dir, current_row, current_col
                )
                next_global_index = global_state.get_global_cell_index(
                    tile_row, tile_col, current_row, current_col
                )
                if (
                    next_global_index in global_acc
                    and global_acc[next_global_index] != 0
                ):
                    # Flow reaches a cell accounted for in global accumulation
                    break

    return flow_acc, tile_row, tile_col


def flow_accumulation_tiled(
    input_path: str,
    output_path: str,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Compute flow accumulation using a tiled approach for large rasters.

    This function orchestrates the tiled flow accumulation process, including:
    1. Setting up the working environment
    2. Processing individual tiles
    3. Calculating global accumulation
    4. Finalizing tile accumulations based on global data

    Args:
        input_path (str): Path to the input flow direction raster.
        output_path (str): Path for the output flow accumulation raster.
        chunk_size (int): Size of each tile (chunk) in pixels.
        progress_callback (ProgressCallback | None): Optional callback for progress updates.
            If None, the operation runs silently.

    Returns:
        None
    """
    # Setup progress tracking
    if progress_callback is None:
        progress_callback = silent_callback
    tracker = ProgressTracker(progress_callback, "Flow Accumulation", total_steps=3)

    # Setup datasets
    flow_dir_ds, output_ds, no_data_value = setup_datasets(input_path, output_path)
    global_state, input_band, output_band = init_global_state(
        flow_dir_ds, output_ds, no_data_value, chunk_size
    )

    # Setup for parallel processing
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    def handle_flow_acc_tile_result(future):
        """Handle the result of a single tile flow accumulation calculation."""
        (
            flow_acc,
            flow_acc_perimeter,
            flow_dir_perimeter,
            links_row_perimeter,
            links_col_perimeter,
            tile_row,
            tile_col,
        ) = future.result()
        with lock:
            global_state.update_perimeters(
                tile_row,
                tile_col,
                flow_acc_perimeter,
                flow_dir_perimeter,
                links_row_perimeter,
                links_col_perimeter,
            )
            flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            flow_acc_tile.from_numpy(flow_acc)
            flow_acc_tile.write(output_band)
            task_queue.get()

    tracker.update(1, "Calculating flow accumulation for each tile")

    # Process each tile in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_dir_tile in raster_chunker(
            input_band, chunk_size, progress_callback=progress_callback
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            future = executor.submit(
                calculate_flow_accumulation_tile,
                flow_dir_tile.data,
                flow_dir_tile.row,
                flow_dir_tile.col,
            )
            future.add_done_callback(handle_flow_acc_tile_result)

    # Wait for all tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()

    # Calculate global accumulation
    tracker.update(2, "Calculating global accumulation")
    global_acc, global_offset = global_state.calculate_global_accumulation()

    def handle_finalize_flow_acc_result(future):
        """Handle the result of finalizing flow accumulation for a tile."""
        with lock:
            try:
                flow_acc, tile_row, tile_col = future.result()
                flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
                flow_acc_tile.from_numpy(flow_acc)
                flow_acc_tile.write(output_band)
            except Exception as e:
                print("Warning: Error finalizing flow accumulation for a tile", e)
            finally:
                task_queue.get()

    tracker.update(3, "Finalizing flow accumulation for each tile")

    # Finalize flow accumulation for each tile in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_acc_tile in raster_chunker(
            output_band, chunk_size, lock=lock, progress_callback=progress_callback
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            flow_dir_tile = RasterChunk(
                flow_acc_tile.row, flow_acc_tile.col, chunk_size, 0
            )
            flow_dir_tile.read(input_band)

            future = executor.submit(
                finalize_flow_accumulation,
                flow_acc_tile.data,
                flow_dir_tile.data,
                global_acc,
                global_offset,
                flow_acc_tile.row,
                flow_acc_tile.col,
                global_state,
            )
            future.add_done_callback(handle_finalize_flow_acc_result)

    # Wait for all finalization tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()
    output_ds = None


def setup_datasets(input_path, output_path):
    """
    Set up input and output datasets for flow accumulation calculation.

    Args:
        input_path (str): Path to input flow direction raster.
        output_path (str): Path for output flow accumulation raster.

    Returns:
        tuple: Flow direction dataset, output dataset, and no data value.
    """
    flow_dir_ds = open_dataset(input_path)
    input_band = flow_dir_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
    if no_data_value is None:
        raise ValueError("Input raster must have a no data value")
    geotransform = flow_dir_ds.GetGeoTransform()
    projection = flow_dir_ds.GetProjection()
    output_ds = create_dataset(
        output_path,
        FLOW_ACCUMULATION_NODATA,
        gdal.GDT_Int64,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    return flow_dir_ds, output_ds, no_data_value


def init_global_state(flow_dir_ds, output_ds, no_data_value, chunk_size):
    """
    Initialize the global state for tiled flow accumulation processing.

    Args:
        flow_dir_ds (gdal.Dataset): Flow direction dataset.
        output_ds (gdal.Dataset): Output dataset.
        no_data_value (float): No data value for the raster.
        chunk_size (int): Size of each tile (chunk) in pixels.

    Returns:
        tuple: Global state object, input band, and output band.
    """
    input_band = flow_dir_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    n_chunks_row = math.ceil(input_band.YSize / chunk_size)
    n_chunks_col = math.ceil(input_band.XSize / chunk_size)
    global_state = GlobalState(n_chunks_row, n_chunks_col, chunk_size, no_data_value)
    return global_state, input_band, output_band
