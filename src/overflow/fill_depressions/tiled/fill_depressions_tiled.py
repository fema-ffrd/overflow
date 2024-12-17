import tempfile
import os
import concurrent.futures
import time
from threading import Lock
import queue
import shutil
import math
import numpy as np
from rich.console import Console
from numba import njit, prange
import numba
from osgeo import gdal
from overflow.util.raster import open_dataset, create_dataset
from overflow.fill_depressions.core import priority_flood_tile, make_sides
from overflow.util.raster import RasterChunk, raster_chunker
from overflow.util.perimeter import get_tile_perimeter
from .global_state import GlobalState

# temporary file for storing labels
LABELS_FILENAME = "labels.tif"


def setup_working_dir(working_dir: str | None):
    """
    Setup working directory for filling depressions in a digital elevation model (DEM).
    """
    cleanup_working_dir = False
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
        cleanup_working_dir = True
    return working_dir, cleanup_working_dir


def setup_datasets(
    input_path: str, output_path: str, working_dir: str
) -> tuple[gdal.Dataset, gdal.Dataset, gdal.Dataset, float]:
    """
    Setup input and output datasets for filling depressions in a digital elevation model (DEM).
    """
    dem_ds = open_dataset(
        input_path, gdal.GA_Update if output_path is None else gdal.GA_ReadOnly
    )
    input_band = dem_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
    if no_data_value is None:
        raise ValueError("Input raster must have a no data value")
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    if output_path is None:
        output_ds = dem_ds
    else:
        output_ds = create_dataset(
            output_path,
            no_data_value,
            input_band.DataType,
            input_band.XSize,
            input_band.YSize,
            geotransform,
            projection,
        )
    labels_ds = create_dataset(
        os.path.join(working_dir, LABELS_FILENAME),
        0,
        gdal.GDT_Int64,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    return dem_ds, output_ds, labels_ds, no_data_value


def init_global_state(
    dem_ds: gdal.Dataset,
    output_ds: gdal.Dataset,
    labels_ds: gdal.Dataset,
    no_data_value: float,
    chunk_size: int,
) -> tuple[GlobalState, gdal.Band, gdal.Band, gdal.Band]:
    """
    Initialize global state for filling depressions in a digital elevation model (DEM).
    """
    input_band = dem_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    labels_band = labels_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    n_chunks_row = math.ceil(input_band.YSize / chunk_size)
    n_chunks_col = math.ceil(input_band.XSize / chunk_size)
    global_state = GlobalState(n_chunks_row, n_chunks_col, chunk_size, no_data_value)
    return global_state, input_band, output_band, labels_band


@njit(nogil=True)
def fill_tile(
    dem: np.ndarray,
    tile_row: int,
    tile_col: int,
    global_state: GlobalState,
    fill_holes: bool,
):
    """
    Fill depressions in a tile of a digital elevation model (DEM).
    This will be called in parallel for each tile and update the global state.
    """
    top = tile_row == 0
    right = tile_col == global_state.num_cols - 1
    bottom = tile_row == global_state.num_rows - 1
    left = tile_col == 0
    sides = make_sides(top, right, bottom, left)
    tile_index = np.int64(tile_row) * np.int64(global_state.num_cols) + np.int64(
        tile_col
    )
    chunk_size_squared = np.int64(global_state.chunk_size) * np.int64(
        global_state.chunk_size
    )
    label_offset = chunk_size_squared * tile_index + np.int64(2)
    labels, graph, _ = priority_flood_tile(
        dem, sides, global_state.no_data, label_offset, fill_holes
    )
    global_state.label_perimeters[tile_index] = get_tile_perimeter(labels)
    global_state.elevation_perimeters[tile_index] = get_tile_perimeter(dem)
    return labels, dem, graph, tile_row, tile_col


@njit(nogil=True, parallel=True)
def raise_tile(
    dem: np.ndarray,
    labels: np.ndarray,
    label_min_elevations: dict[int, float],
    tile_row: int,
    tile_col: int,
    no_data_value: float,
):
    """
    Raise elevation of dem to match global labels.
    This will be called in parallel for each tile.
    """
    n_row, n_col = dem.shape
    for row in prange(n_row):  # pylint: disable=not-an-iterable
        for col in range(n_col):
            height = dem[row, col]
            label = labels[row, col]
            if (
                label != 0
                and dem[row, col] != no_data_value
                and not np.isnan(dem[row, col])
            ):
                dem[row, col] = max(label_min_elevations[label], height)
    return dem, tile_row, tile_col


def fill_depressions_tiled(
    input_path, output_path, chunk_size, working_dir, fill_holes=False
):
    """
    Fill depressions in a digital elevation model (DEM) using a parallel tiled approach.

    This function implements the parallel depression-filling algorithm described in the paper
    "Parallel Priority-Flood Depression Filling For Trillion Cell Digital Elevation Models On
    Desktops Or Clusters" by Richard Barnes (2016). It divides the input DEM into tiles,
    processes each tile independently using multiple threads, and then connects the tiles to
    obtain a depression-filled DEM.

    Parameters:
        input_path (str): Path to the input DEM file.
        output_path (str): Path to the output depression-filled DEM file.
        chunk_size (int): Size of each tile in pixels.
        working_dir (str): Directory to store temporary files during processing.
        fill_holes (bool): If True, fills holes in the DEM before filling depressions.

    Returns:
        None

    The function performs the following steps:
    1. Setup:
       - Sets up the working directory.
       - Opens the input DEM dataset, creates the output dataset and labels dataset.
       - Initializes the global state with information about the DEM, tile size, and a graph
         to store the relationships between tiles.

    2. Tile Processing:
       - Divides the input DEM into tiles and processes each tile concurrently using multiple
         threads.
       - Each tile is processed using the Priority-Flood algorithm to fill depressions within
         the tile.
       - The results of each tile processing (updated labels, DEM, graph) are stored in the
         global state and written to the corresponding datasets.

    3. Connecting Tiles:
       - Connects the edges and corners of adjacent tiles to ensure proper depression filling
         across tile boundaries.
       - Solves the graph to determine the minimum elevations for each label across all tiles.

    4. Raising Elevations:
       - Raises the elevation of each tile's DEM to match the global minimum elevations
         determined in the previous step.
       - Updates the output DEM dataset with the raised elevations.

    5. Teardown:
       - Cleans up the working directory if specified and closes the datasets.

    Note:
    - The function utilizes parallel processing by dividing the DEM into tiles and processing
      them concurrently using multiple threads. Numba's nogil feature is used to release the
      Global Interpreter Lock (GIL) and allow parallel execution of Numba-compiled functions.
    - The tile size (chunk_size) can be adjusted based on the available memory and
      computational resources.
    - The function follows the overall structure of Algorithm 3 in the paper, which describes
      the main steps of the parallel depression-filling algorithm.
    """
    console = Console()
    # setup
    working_dir, cleanup_working_dir = setup_working_dir(working_dir)
    dem_ds, output_ds, labels_ds, no_data_value = setup_datasets(
        input_path, output_path, working_dir
    )
    global_state, input_band, output_band, labels_band = init_global_state(
        dem_ds, output_ds, labels_ds, no_data_value, chunk_size
    )

    # Threading setup
    max_workers = numba.config.NUMBA_NUM_THREADS  # pylint: disable=no-member
    task_queue = queue.Queue(max_workers)
    lock = Lock()

    # Fill depressions in each tile
    def handle_fill_tile_result(future):
        labels, dem, graph, tile_row, tile_col = future.result()
        with lock:
            global_state.graph.update(graph)
            labels_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            labels_tile.from_numpy(labels)
            labels_tile.write(labels_band)
            dem_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            dem_tile.from_numpy(dem)
            dem_tile.write(output_band)
            task_queue.get()

    print("Step 1 of 2: Fill depressions in each tile")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(input_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            future = executor.submit(
                fill_tile,
                dem_tile.data,
                dem_tile.row,
                dem_tile.col,
                global_state,
                fill_holes,
            )
            future.add_done_callback(handle_fill_tile_result)

    # wait for all tasks to finish
    while not task_queue.empty():
        time.sleep(0.1)

    # flush cache between writing and reading
    labels_band.FlushCache()
    labels_ds.FlushCache()
    output_band.FlushCache()
    output_ds.FlushCache()

    # connect tile edges and corners and solve the graph
    with console.status("Connecting tiles and solving graph..."):
        global_state.connect_tile_edges_and_corners()
        label_min_elevations = global_state.solve_graph()

    # raise elevation of dem to match global labels
    def handle_raise_tile_result(future):
        with lock:
            dem, tile_row, tile_col = future.result()
            dem_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            dem_tile.from_numpy(dem)
            dem_tile.write(output_band)
            task_queue.get()

    print("Step 2 of 2: Raise elevation of each tile to match global labels")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(output_band, chunk_size, lock=lock):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            tile_labels = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
            tile_labels.read(labels_band)
            future = executor.submit(
                raise_tile,
                dem_tile.data,
                tile_labels.data,
                label_min_elevations,
                dem_tile.row,
                dem_tile.col,
                global_state.no_data,
            )
            future.add_done_callback(handle_raise_tile_result)

    while not task_queue.empty():
        time.sleep(0.1)

    # tear down
    if cleanup_working_dir:
        shutil.rmtree(working_dir)
    output_ds = None
    labels_ds = None
