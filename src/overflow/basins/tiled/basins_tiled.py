import concurrent.futures
import math
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.basins.core import label_watersheds
from overflow.basins.core.basin_polygons import create_basin_polygons
from overflow.util.constants import DEFAULT_CHUNK_SIZE
from overflow.util.perimeter import get_tile_perimeter
from overflow.util.progress import ProgressCallback, ProgressTracker, silent_callback
from overflow.util.raster import (
    RasterChunk,
    create_dataset,
    raster_chunker,
)

from .global_state import GlobalState


@njit(nogil=True)
def finalize_watersheds(
    labels: np.ndarray,
    global_graph: dict,
    drainage_points: dict,
    tile_row: int,
    tile_col: int,
    all_basins=True,
):
    """
    This will update the labels based on the global graph and drainage points.
    Assigns the final watershed ID to each cell in the raster which will either
    be the first downstream drainage point or an outlet for the entire raster.

    Parameters:
    - labels (np.ndarray): The labeled array.
    - global_graph (dict): The global graph representing the watershed connections.
    - drainage_points (dict): A dictionary mapping drainage point coordinates to their labels.
    - all_basins (bool): If True, label all basins. If False, only label basins connected to drainage points.

    Description:
    This function iterates over each element in the labeled array and updates the labels
    based on the global graph and drainage points. For each label that is not a drainage point,
    it walks down the graph until it finds a node that is either a drainage point or not in the graph.
    The label is then updated with the final node label.

    Note:
     - This function modifies the labels array in place.
     - The resulting labels will be either the first downstream drainage point or an outlet
       for the entire raster.
    """
    # dp labels are the set of values in the drainage points
    drainage_point_labels = set(drainage_points.values())

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            current_label = labels[row, col]
            if current_label not in drainage_point_labels:
                # walk down the graph until we find a node that is in the drainage points
                # or we reach the end of the graph (i.e. the node is not in the graph)
                while (
                    current_label in global_graph
                    and current_label not in drainage_point_labels
                ):
                    current_label = global_graph[current_label]
                if current_label in drainage_point_labels or all_basins:
                    labels[row, col] = current_label
                else:
                    # if we reach the end of the graph, the node is an outlet
                    labels[row, col] = 0
    return labels, tile_row, tile_col


@njit(nogil=True)
def process_tile(
    fdr: np.ndarray,
    drainage_points: dict,
    id_offset: int,
    row_offset: int,
    col_offset: int,
    tile_row: int,
    tile_col: int,
    tile_index: int,
) -> tuple:
    """
    Helper function to process a single tile.
    """
    watersheds, local_graph = label_watersheds(
        fdr, drainage_points, id_offset, row_offset, col_offset
    )
    fdr_perimeter = get_tile_perimeter(fdr)
    return watersheds, local_graph, fdr_perimeter, tile_row, tile_col, tile_index


def label_watersheds_tiled(
    fdr_filepath: str,
    drainage_points: dict,
    output_filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    all_basins: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """
    Label watersheds using a tiled approach.

    Parameters:
    - fdr_filepath (str): The file path to the flow direction raster.
    - drainage_points (dict): A dictionary mapping drainage point coordinates to their labels.
    - output_filepath (str): The file path to save the labeled watersheds raster.
    - chunk_size (int): The size of each chunk (default: DEFAULT_CHUNK_SIZE).
    - all_basins (bool): If True, label all basins. If False, only label basins connected to drainage points.
    - progress_callback (ProgressCallback | None): Optional callback for progress updates.
        If None, the operation runs silently.

    Returns:
    - dict: The global graph representing the watershed connections. This is a singly
            linked list where each key is a watershed ID and the value is the downstream
            watershed ID.

    Description:
    This function labels watersheds in a tiled manner. It reads the flow direction raster in chunks,
    labels the watersheds for each chunk, and updates the global state with the tile information.
    It then completes the global graph by connecting the watersheds across tile boundaries.
    Finally, it applies the watershed map to assign basin labels to drainage points or outlets.
    The labeled watersheds are saved to the specified output file path.

    This function is designed to handle large rasters by processing them in chunks.
    """
    # Setup progress tracking
    if progress_callback is None:
        progress_callback = silent_callback
    tracker = ProgressTracker(progress_callback, "Label Watersheds", total_steps=3)
    fdr_ds = gdal.Open(fdr_filepath)
    fdr_band = fdr_ds.GetRasterBand(1)
    gt = fdr_ds.GetGeoTransform()
    projection = fdr_ds.GetProjection()

    labels_ds = create_dataset(
        output_filepath,
        0,
        gdal.GDT_Int64,
        fdr_band.XSize,
        fdr_band.YSize,
        gt,
        projection,
    )
    labels_band = labels_ds.GetRasterBand(1)

    tile_rows = math.ceil(fdr_band.YSize / chunk_size)
    tile_cols = math.ceil(fdr_band.XSize / chunk_size)
    global_state = GlobalState(tile_rows, tile_cols, chunk_size)
    tile_index = 0

    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)

    lock = Lock()

    def handle_result(future):
        with lock:
            watersheds, local_graph, fdr_perimeter, tile_row, tile_col, tile_index = (
                future.result()
            )
            labels_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            labels_tile.from_numpy(watersheds)
            labels_tile.write(labels_band)
            watersheds_perimeter = get_tile_perimeter(watersheds)
            global_state.flow_directions[tile_index] = fdr_perimeter
            global_state.watersheds[tile_index] = watersheds_perimeter
            global_state.graph.update(local_graph)
            task_queue.get()

    tracker.update(1, "Labeling tiles")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fdr_tile in raster_chunker(
            fdr_band, chunk_size, 0, progress_callback=progress_callback
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(tile_index)
            tile_row = fdr_tile.row
            tile_col = fdr_tile.col
            row_offset = tile_row * chunk_size
            col_offset = tile_col * chunk_size
            id_offset = tile_index * chunk_size * chunk_size
            future = executor.submit(
                process_tile,
                fdr_tile.data,
                drainage_points,
                id_offset,
                row_offset,
                col_offset,
                tile_row,
                tile_col,
                tile_index,
            )
            future.add_done_callback(handle_result)
            tile_index += 1

    # wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    labels_band.FlushCache()
    labels_ds.FlushCache()
    fdr_ds = None

    global_state.complete_graph()

    def handle_finalized_tile(future):
        with lock:
            finalized_labels, row, col = future.result()
            chunk = RasterChunk(row, col, chunk_size, 0)
            chunk.from_numpy(finalized_labels)
            chunk.write(labels_band)
            task_queue.get()

    tracker.update(2, "Finalizing watersheds")

    # iterate over labels now using the graph to assign ids to drainage points or outlets
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for labels_tile in raster_chunker(
            labels_band, chunk_size, lock=lock, progress_callback=progress_callback
        ):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(1)
            future = executor.submit(
                finalize_watersheds,
                labels_tile.data,
                global_state.graph,
                drainage_points,
                labels_tile.row,
                labels_tile.col,
                all_basins,
            )
            future.add_done_callback(handle_finalized_tile)

    # Wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    labels_band.FlushCache()

    graph = global_state.graph

    tracker.update(3, "Creating basin polygons")

    basin_polygons_filepath = output_filepath.replace(".tif", ".gpkg")
    create_basin_polygons(
        labels_band,
        graph,
        chunk_size,
        basin_polygons_filepath,
        gt,
        projection,
        progress_callback,
    )

    labels_ds = None
    return graph
