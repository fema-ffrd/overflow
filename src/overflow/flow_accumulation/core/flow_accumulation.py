import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_EXTERNAL,
    FLOW_TERMINATES,
    NEIGHBOR_OFFSETS,
)
from overflow.util.queue import Int64PairQueue as Queue
from overflow.util.raster import create_dataset


@njit
def get_next_cell(
    flow_direction: np.ndarray, row: int, col: int
) -> tuple[int, int, int]:
    """Return the next (downstream) row, column, and value in the flow direction raster.

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): The row of the current cell
        col (int): The col of the current cell

    Returns:
        tuple[int, int, int]: the (row, col, val) of the next (downstream) cell
    """
    fdr_value = flow_direction[row, col]
    # Safety check: prevent out-of-bounds access on NEIGHBOR_OFFSETS (length 8)
    # FLOW_DIRECTION_UNDEFINED = 8, FLOW_DIRECTION_NODATA = 9
    if fdr_value >= 8:
        return row, col, fdr_value
    d_row, d_col = NEIGHBOR_OFFSETS[fdr_value]
    next_row = row + d_row
    next_col = col + d_col
    rows, cols = flow_direction.shape
    is_outside_tile = (
        next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
    )
    if not is_outside_tile:
        return next_row, next_col, flow_direction[next_row, next_col]
    return next_row, next_col, FLOW_DIRECTION_NODATA


@njit
def perimeter_indices(shape):
    """Return the indices of the perimeter cells of a 2D array.

    Parameters
    ----------
    shape : tuple
        The shape (rows, cols) of the array.

    Returns
    -------
    list
        List of (row, col) tuples representing perimeter indices.
    """
    rows, cols = shape
    indices = []
    for i in range(rows):
        indices.append((i, 0))
        indices.append((i, cols - 1))
    for j in range(1, cols - 1):
        indices.append((0, j))
        indices.append((rows - 1, j))
    return indices


@njit
def follow_path(flow_direction, row, col, links):
    """Follow the flow path from a cell on the perimeter of the tile and
    populate the links array with the row and col of the perimeter cell that each cell drains to.
    If the cell drains directly to the edge of the tile, the FLOW_EXTERNAL value is used.
    If the cell terminates within the tile, the FLOW_TERMINATES value is used.
    If the cell drains through the tile, the row and col of the perimeter cell it drains to are used.
    This is Algorithm 2 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): the row of the cell on the perimeter
        col (int): the col of the cell on the perimeter
        links (np.ndarray): 3D array of shape (rows, cols, 2)
    """
    init_row = row
    init_col = col
    rows, cols = flow_direction.shape
    # Cycle detection: no valid path should exceed total cells in tile
    max_iterations = rows * cols
    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations:
            # Cycle detected - mark as terminating to prevent infinite loop
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            print("Warning: Cycle detected in flow direction data.")
            break
        next_row, next_col, next_val = get_next_cell(flow_direction, row, col)
        is_outside_tile = (
            next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
        )
        if is_outside_tile:
            if row == init_row and col == init_col:
                # cell drains immediately to the edge of the tile
                links[init_row, init_col, 0] = FLOW_EXTERNAL[0]
                links[init_row, init_col, 1] = FLOW_EXTERNAL[1]
            else:
                # cell drains through the tile
                links[init_row, init_col, 0] = row
                links[init_row, init_col, 1] = col
            break
        if next_val in (FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED):
            # cell terminates within the tile
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            break
        row, col = next_row, next_col


@njit
def single_tile_flow_accumulation(
    flow_direction: np.ndarray, create_links: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate flow accumulation for a single tile.
       This is Algorithm 1 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster with codes from FlowDirection constants

    Returns:
        tuple[np.ndarray, np.ndarray]: (flow_accumulation, links)
         - flow_accumulation: The flow accumulation raster
         - links: The link raster identifying where each perimeter cell drains to
    """
    # the output flow accumulation raster
    flow_accumulation = np.zeros_like(flow_direction, dtype=np.int64)
    # a dependency raster (number of neighbors that flow into each cell)
    inflow_count = np.zeros_like(flow_direction, dtype=np.uint8)

    # Calculate inflow count
    for row, col in np.ndindex(flow_direction.shape):
        value = flow_direction[row, col]
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if value == FLOW_DIRECTION_NODATA:
            flow_accumulation[row, col] = FLOW_ACCUMULATION_NODATA
            continue
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        inflow_count[next_row, next_col] += 1

    queue = Queue([(0, 0)])
    queue.pop()

    # Populate initial queue with cells that have no inflow
    for row, col in np.ndindex(inflow_count.shape):
        _, _, next_value = get_next_cell(flow_direction, row, col)
        if (
            inflow_count[row, col] == 0
            and flow_accumulation[row, col] != FLOW_ACCUMULATION_NODATA
        ):
            queue.push((row, col))

    # main loop
    while queue:
        row, col = queue.pop()
        flow_accumulation[row, col] += 1
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        flow_accumulation[next_row, next_col] += flow_accumulation[row, col]
        inflow_count[next_row, next_col] -= 1
        if inflow_count[next_row, next_col] == 0:
            queue.push((next_row, next_col))

    if not create_links:
        return flow_accumulation, None  # type: ignore[return-value]
    # links is a 3d numpy array containing the row and col on the
    # perimeter that each cell utlimatly drains to.
    # Only cells on the perimeter of the tile are considered.
    # Cells that drain directly to the edge of the tile are
    # marked with FLOW_EXTERNAL. Cells that terminate within the tile
    # are marked with FLOW_TERMINATES. Cells that drain through the tile
    # are marked with the row and col of the perimeter cell they drain to.
    links = np.empty(
        shape=(flow_direction.shape[0], flow_direction.shape[1], 2), dtype=np.int64
    )
    for row, col in perimeter_indices(flow_direction.shape):
        follow_path(flow_direction, row, col, links)
    return flow_accumulation, links


def flow_accumulation(fdr_path: str, output_fac_path: str) -> None:
    """
    Generates a flow accumulation raster from a flow direction raster.

    Parameters
    ----------
    fdr_path : str
        Path to the input flow direction raster file.
    output_fac_path : str
        Path to the output flow accumulation raster file.

    Returns
    -------
    None
    """
    fdr_ds = gdal.Open(fdr_path)
    projection = fdr_ds.GetProjection()
    transform = fdr_ds.GetGeoTransform()

    fdr_ds_band = fdr_ds.GetRasterBand(1)
    fdr_array = fdr_ds_band.ReadAsArray().astype("uint8")
    fac_ds = create_dataset(
        output_fac_path,
        FLOW_ACCUMULATION_NODATA,
        gdal.GDT_Int64,
        fdr_ds.RasterXSize,
        fdr_ds.RasterYSize,
        transform,
        projection,
    )
    fac_band = fac_ds.GetRasterBand(1)
    fac_array, _ = single_tile_flow_accumulation(fdr_array, False)
    fac_band.WriteArray(fac_array)
    fac_band.FlushCache()
    fac_ds.FlushCache()
    fac_band = None
    fac_ds = None
    fdr_ds_band = None
    fdr_ds = None
