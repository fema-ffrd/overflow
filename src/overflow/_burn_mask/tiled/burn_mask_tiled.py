import concurrent.futures
import math
import os
import queue
import shutil
import tempfile
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow._burn_mask.core import (
    MaskSelection,
    accumulate_region_stats,
    apply_burn,
    burn_direct,
    label_regions,
)
from overflow._util.constants import BURN_METHOD_STATISTIC, BURN_NO_REGION_LABEL
from overflow._util.perimeter import get_tile_perimeter
from overflow._util.progress import ProgressCallback, ProgressTracker, silent_callback
from overflow._util.raster import (
    RasterChunk,
    create_dataset,
    open_dataset,
    raster_chunker,
)

from .global_state import GlobalState

# temporary file for storing region labels
LABELS_FILENAME = "burn_labels.tif"


def setup_working_dir(working_dir: str | None) -> tuple[str, bool]:
    """
    Setup working directory for burning mask regions into a DEM.
    """
    cleanup_working_dir = False
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
        cleanup_working_dir = True
    return working_dir, cleanup_working_dir


def setup_datasets(
    dem_path: str, mask_path: str, output_path: str
) -> tuple[gdal.Dataset, gdal.Dataset, gdal.Dataset, float, float | None]:
    """
    Open the input DEM and mask and create the output DEM.

    Returns the three datasets, the DEM's nodata value and the mask's nodata value
    (which may be None, since binary masks commonly do not declare one).
    """
    dem_ds = open_dataset(dem_path)
    mask_ds = open_dataset(mask_path)
    dem_band = dem_ds.GetRasterBand(1)
    dem_nodata = dem_band.GetNoDataValue()
    if dem_nodata is None:
        raise ValueError("Input DEM must have a no data value")
    mask_nodata = mask_ds.GetRasterBand(1).GetNoDataValue()
    output_ds = create_dataset(
        output_path,
        dem_nodata,
        dem_band.DataType,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    return dem_ds, mask_ds, output_ds, dem_nodata, mask_nodata


def tile_valid_extent(
    tile_row: int, tile_col: int, chunk_size: int, band: gdal.Band
) -> tuple[int, int]:
    """
    Return how many rows and columns of a tile hold real raster data.

    Tiles hanging off the right or bottom edge of the raster are padded out to a
    full square by the reader, and those padded cells must be excluded from region
    labeling.
    """
    valid_rows = min(chunk_size, max(band.YSize - tile_row * chunk_size, 0))
    valid_cols = min(chunk_size, max(band.XSize - tile_col * chunk_size, 0))
    return valid_rows, valid_cols


@njit(nogil=True)
def label_tile(
    dem: np.ndarray,
    mask: np.ndarray,
    selection: MaskSelection,
    neighbor_offsets: np.ndarray,
    dem_nodata: float,
    valid_rows: int,
    valid_cols: int,
    tile_row: int,
    tile_col: int,
    global_state: GlobalState,
) -> tuple:
    """
    Label the mask regions in one tile and accumulate their DEM statistics.

    Labels are drawn from a range private to this tile, so they are globally unique
    without a renumbering pass. The tile's label and mask perimeters are stashed in
    the global state; they are all the later solve needs to discover which of these
    labels are really the same region.

    This is called in parallel for each tile. Each tile writes only its own row of
    the perimeter arrays, so no lock is needed here; the statistics are merged into
    the global dicts by the caller under a lock instead.
    """
    tile_index = np.int64(tile_row) * np.int64(global_state.num_cols) + np.int64(
        tile_col
    )
    chunk_size_squared = np.int64(global_state.chunk_size) * np.int64(
        global_state.chunk_size
    )
    label_offset = chunk_size_squared * tile_index + np.int64(BURN_NO_REGION_LABEL + 1)
    labels, region_count = label_regions(
        mask, selection, label_offset, neighbor_offsets, valid_rows, valid_cols
    )
    stats = accumulate_region_stats(dem, labels, label_offset, region_count, dem_nodata)
    global_state.label_perimeters[tile_index] = get_tile_perimeter(labels)
    global_state.mask_perimeters[tile_index] = get_tile_perimeter(mask).astype(
        np.float64
    )
    return labels, stats, label_offset, tile_row, tile_col


def _burn_direct_tiled(
    dem_ds: gdal.Dataset,
    mask_ds: gdal.Dataset,
    output_ds: gdal.Dataset,
    method: int,
    selection: MaskSelection,
    dem_nodata: float,
    mask_fill: float | None,
    chunk_size: int,
    tracker: ProgressTracker,
) -> None:
    """
    Burn constant or relative values in a single tiled pass.

    Neither method depends on anything outside the cell being written, so there is
    no labeling, no temporary raster and no cross tile reconciliation.
    """
    dem_band = dem_ds.GetRasterBand(1)
    mask_band = mask_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)

    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()
    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = math.ceil(dem_band.YSize / chunk_size) * math.ceil(
        dem_band.XSize / chunk_size
    )

    def handle_burn_tile_result(future):
        dem, tile_row, tile_col = future.result()
        with lock:
            dem_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            dem_tile.from_numpy(dem)
            dem_tile.write(output_band)
            task_queue.get()
            chunk_counter[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    tracker.update(1, step_name="Burn regions")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(dem_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            mask_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
            mask_tile.read(mask_band, mask_fill)
            future = executor.submit(
                burn_direct,
                dem_tile.data,
                mask_tile.data,
                selection,
                method,
                dem_nodata,
                dem_tile.row,
                dem_tile.col,
            )
            future.add_done_callback(handle_burn_tile_result)

    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()


def _burn_statistic_tiled(
    dem_ds: gdal.Dataset,
    mask_ds: gdal.Dataset,
    output_ds: gdal.Dataset,
    selection: MaskSelection,
    statistic: int,
    burn_offset: float,
    neighbor_offsets: np.ndarray,
    connectivity: int,
    dem_nodata: float,
    mask_fill: float | None,
    chunk_size: int,
    working_dir: str,
    tracker: ProgressTracker,
) -> None:
    """
    Burn a per region statistic in three passes.

    1. Label each tile's regions independently and accumulate min, max, sum and
       count of the DEM under each label.
    2. Walk the tile seams, join labels that turn out to belong to the same region,
       and merge their statistics. This is what makes a region that straddles two
       or more tiles come out with one statistic rather than one per fragment.
    3. Re-read the original DEM and write each region's solved value.
    """
    dem_band = dem_ds.GetRasterBand(1)
    mask_band = mask_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)

    labels_ds = create_dataset(
        os.path.join(working_dir, LABELS_FILENAME),
        BURN_NO_REGION_LABEL,
        gdal.GDT_Int64,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    labels_band = labels_ds.GetRasterBand(1)

    n_chunks_row = math.ceil(dem_band.YSize / chunk_size)
    n_chunks_col = math.ceil(dem_band.XSize / chunk_size)
    global_state = GlobalState(n_chunks_row, n_chunks_col, chunk_size, connectivity)

    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()
    total_chunks = n_chunks_row * n_chunks_col

    # pass 1: label regions and accumulate statistics per tile
    chunk_counter = [0]  # Use list for mutability in closure

    def handle_label_tile_result(future):
        labels, stats, label_offset, tile_row, tile_col = future.result()
        with lock:
            global_state.merge_tile_stats(label_offset, stats)
            labels_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            labels_tile.from_numpy(labels)
            labels_tile.write(labels_band)
            task_queue.get()
            chunk_counter[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    tracker.update(1, step_name="Label regions")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(dem_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            mask_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
            mask_tile.read(mask_band, mask_fill)
            valid_rows, valid_cols = tile_valid_extent(
                dem_tile.row, dem_tile.col, chunk_size, dem_band
            )
            future = executor.submit(
                label_tile,
                dem_tile.data,
                mask_tile.data,
                selection,
                neighbor_offsets,
                dem_nodata,
                valid_rows,
                valid_cols,
                dem_tile.row,
                dem_tile.col,
                global_state,
            )
            future.add_done_callback(handle_label_tile_result)

    while not task_queue.empty():
        time.sleep(0.1)

    # flush cache between writing and reading
    labels_band.FlushCache()
    labels_ds.FlushCache()

    # pass 2: join regions that span tiles and reduce them to burn values
    tracker.update(2, step_name="Solve regions")
    global_state.connect_tile_edges_and_corners()
    burn_lookup = global_state.solve_regions(statistic, burn_offset)

    # pass 3: write the solved value for each region
    chunk_counter_apply = [0]  # Use list for mutability in closure

    def handle_apply_tile_result(future):
        dem, tile_row, tile_col = future.result()
        with lock:
            dem_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            dem_tile.from_numpy(dem)
            dem_tile.write(output_band)
            task_queue.get()
            chunk_counter_apply[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter_apply[0]}/{total_chunks}")

    tracker.update(3, step_name="Apply burn values")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(dem_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            labels_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
            labels_tile.read(labels_band)
            future = executor.submit(
                apply_burn,
                dem_tile.data,
                labels_tile.data,
                burn_lookup,
                dem_nodata,
                dem_tile.row,
                dem_tile.col,
            )
            future.add_done_callback(handle_apply_tile_result)

    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()
    labels_ds.Close()


def _burn_mask_tiled(
    dem_path: str,
    mask_path: str,
    output_path: str,
    method: int,
    selection: MaskSelection,
    statistic: int,
    burn_offset: float,
    neighbor_offsets: np.ndarray,
    connectivity: int,
    chunk_size: int,
    working_dir: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Burn mask regions into a DEM using a parallel tiled approach.

    The input DEM is divided into tiles that are processed concurrently by numba
    kernels compiled with nogil, following the same threading pattern as the other
    tiled tools in this package.

    The constant and relative methods are per cell functions of the DEM and mask
    values, so they run in a single pass. The statistic method needs each contiguous
    region's statistic, which a single tile cannot know for a region that continues
    into its neighbors, so it runs in three passes: label and accumulate per tile,
    join labels across tile seams and merge their statistics, then apply.

    Parameters
    ----------
    dem_path : str
        Path to the input DEM raster.
    mask_path : str
        Path to the co-registered mask raster.
    output_path : str
        Path to the output burned DEM raster.
    method : int
        One of the BURN_METHOD_* constants.
    selection : MaskSelection
        Which mask values identify regions and, for the constant and relative
        methods, how much to burn for each.
    statistic : int
        One of the BURN_STAT_* constants. Only used by the statistic method.
    burn_offset : float
        Subtracted from each computed statistic. Only used by the statistic method.
    neighbor_offsets : np.ndarray
        Row/column offsets defining region connectivity.
    connectivity : int
        Either 4 or 8, matching neighbor_offsets.
    chunk_size : int
        Size of each tile in cells.
    working_dir : str | None
        Directory for the temporary labels raster used by the statistic method.
        If None, a system temp directory is created and removed afterwards.
    progress_callback : ProgressCallback | None
        Optional callback for progress updates. If None, the operation runs
        silently.

    Returns
    -------
    None
    """
    if progress_callback is None:
        progress_callback = silent_callback
    is_statistic = method == BURN_METHOD_STATISTIC
    tracker = ProgressTracker(
        progress_callback, "Burning mask regions", total_steps=3 if is_statistic else 1
    )

    dem_ds, mask_ds, output_ds, dem_nodata, mask_nodata = setup_datasets(
        dem_path, mask_path, output_path
    )
    # tiles at the right and bottom raster edges are padded out to a full square.
    # The pad value is irrelevant to the result because label_regions ignores cells
    # outside the raster, but the reader still needs something to fill with when the
    # mask declares no nodata value.
    mask_fill = None if mask_nodata is not None else 0.0

    cleanup_working_dir = False
    try:
        if is_statistic:
            working_dir, cleanup_working_dir = setup_working_dir(working_dir)
            _burn_statistic_tiled(
                dem_ds,
                mask_ds,
                output_ds,
                selection,
                statistic,
                burn_offset,
                neighbor_offsets,
                connectivity,
                dem_nodata,
                mask_fill,
                chunk_size,
                working_dir,
                tracker,
            )
        else:
            _burn_direct_tiled(
                dem_ds,
                mask_ds,
                output_ds,
                method,
                selection,
                dem_nodata,
                mask_fill,
                chunk_size,
                tracker,
            )
    finally:
        output_ds.Close()
        mask_ds.Close()
        dem_ds.Close()
        if cleanup_working_dir and working_dir is not None:
            shutil.rmtree(working_dir)
