import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow._util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_EXTERNAL,
    FLOW_TERMINATES,
    NEIGHBOR_OFFSETS,
    WEIGHTED_ACCUMULATION_NODATA,
    WEIGHTS_NODATA_MODE_PROPAGATE,
    WEIGHTS_NODATA_MODE_ZERO,
)
from overflow._util.queue import Int64PairQueue as Queue
from overflow._util.raster import create_dataset


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
def follow_path(flow_direction, row, col, links, tile_row=0, tile_col=0):
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
        tile_row (int): tile row index, used only in warning messages
        tile_col (int): tile col index, used only in warning messages
    """
    init_row = row
    init_col = col
    # sentinel must not collide with out-of-tile coordinates like (-1, -1)
    prev_row = -9999
    prev_col = -9999
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
            print(
                "Warning: Cycle detected in flow direction data. tile:",
                tile_row,
                tile_col,
                "cell:",
                row,
                col,
            )
            break
        next_row, next_col, next_val = get_next_cell(flow_direction, row, col)
        if next_row == prev_row and next_col == prev_col:
            # The current cell and the previous cell point at each other:
            # a two-cell cycle. Terminate immediately instead of spinning
            # until max_iterations.
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            print(
                "Warning: Cycle detected in flow direction data. tile:",
                tile_row,
                tile_col,
                "cell:",
                row,
                col,
            )
            break
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
        prev_row, prev_col = row, col
        row, col = next_row, next_col


@njit
def single_tile_flow_accumulation(
    flow_direction: np.ndarray,
    create_links: bool = True,
    tile_row: int = 0,
    tile_col: int = 0,
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
        follow_path(flow_direction, row, col, links, tile_row, tile_col)
    return flow_accumulation, links


@njit
def single_tile_flow_accumulation_weighted(
    flow_direction: np.ndarray,
    weights: np.ndarray,
    weights_nodata: float,
    nodata_mode: int,
    create_links: bool = True,
    tile_row: int = 0,
    tile_col: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate weighted flow accumulation for a single tile: instead of
       counting cells, sums the weights raster over each cell's upstream
       contributing area. This is Algorithm 1 from
       https://arxiv.org/pdf/1608.04431.pdf R. Barnes, generalized to
       accumulate an arbitrary per-cell weight instead of 1.

    Args:
        flow_direction (np.ndarray): Flow direction raster with codes from FlowDirection constants
        weights (np.ndarray): Per-cell weight raster, same shape as flow_direction
        weights_nodata (float): The nodata value used in the weights raster
        nodata_mode (int): WEIGHTS_NODATA_MODE_ZERO or WEIGHTS_NODATA_MODE_PROPAGATE,
            controlling how nodata weight cells are handled

    Returns:
        tuple[np.ndarray, np.ndarray]: (flow_accumulation, links)
         - flow_accumulation: The weighted flow accumulation raster (float64)
         - links: The link raster identifying where each perimeter cell drains to
    """
    # substitute nodata weight cells with 0.0 (contribute nothing) or NaN
    # (poison this cell and everything downstream of it), once per tile,
    # so the main loop below stays a simple, branch-free `+=`
    effective_weights = np.empty(weights.shape, dtype=np.float64)
    for row, col in np.ndindex(weights.shape):
        value = weights[row, col]
        if value == weights_nodata:
            if nodata_mode == WEIGHTS_NODATA_MODE_PROPAGATE:
                effective_weights[row, col] = np.nan
            else:
                effective_weights[row, col] = 0.0
        else:
            effective_weights[row, col] = value

    # the output flow accumulation raster
    flow_accumulation = np.zeros_like(flow_direction, dtype=np.float64)
    # a dependency raster (number of neighbors that flow into each cell)
    inflow_count = np.zeros_like(flow_direction, dtype=np.uint8)

    # Calculate inflow count
    for row, col in np.ndindex(flow_direction.shape):
        value = flow_direction[row, col]
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if value == FLOW_DIRECTION_NODATA:
            flow_accumulation[row, col] = WEIGHTED_ACCUMULATION_NODATA
            continue
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        inflow_count[next_row, next_col] += 1

    queue = Queue([(0, 0)])
    queue.pop()

    # Populate initial queue with cells that have no inflow
    for row, col in np.ndindex(inflow_count.shape):
        _, _, next_value = get_next_cell(flow_direction, row, col)
        if inflow_count[row, col] == 0 and not np.isnan(flow_accumulation[row, col]):
            queue.push((row, col))

    # main loop
    while queue:
        row, col = queue.pop()
        flow_accumulation[row, col] += effective_weights[row, col]
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
        follow_path(flow_direction, row, col, links, tile_row, tile_col)
    return flow_accumulation, links


def _parse_weights_nodata_mode(mode: str) -> int:
    """Translate the public string weights_nodata_mode API into the internal
    int flags consumed by the njit-compiled weighted accumulation functions.

    Args:
        mode (str): "zero" or "propagate"

    Returns:
        int: WEIGHTS_NODATA_MODE_ZERO or WEIGHTS_NODATA_MODE_PROPAGATE

    Raises:
        ValueError: If mode is not "zero" or "propagate"
    """
    if mode == "zero":
        return WEIGHTS_NODATA_MODE_ZERO
    if mode == "propagate":
        return WEIGHTS_NODATA_MODE_PROPAGATE
    raise ValueError(
        f"Invalid weights_nodata_mode: {mode!r}. Must be 'zero' or 'propagate'."
    )


def _flow_accumulation_weighted(
    fdr_path: str,
    weights_path: str,
    output_fac_path: str,
    weights_nodata_mode: str = "zero",
) -> None:
    """
    Generates a weighted flow accumulation raster from a flow direction raster
    and a co-registered weights raster.

    Parameters
    ----------
    fdr_path : str
        Path to the input flow direction raster file.
    weights_path : str
        Path to the input weights raster file, co-registered with the flow direction raster.
    output_fac_path : str
        Path to the output flow accumulation raster file.
    weights_nodata_mode : str
        "zero" (nodata weight cells contribute 0) or "propagate" (nodata weight
        cells poison their own and all downstream accumulation with NaN).

    Returns
    -------
    None
    """
    nodata_mode = _parse_weights_nodata_mode(weights_nodata_mode)
    fdr_ds = gdal.Open(fdr_path)
    projection = fdr_ds.GetProjection()
    transform = fdr_ds.GetGeoTransform()

    fdr_ds_band = fdr_ds.GetRasterBand(1)
    fdr_array = fdr_ds_band.ReadAsArray().astype("uint8")

    weights_ds = gdal.Open(weights_path)
    weights_band = weights_ds.GetRasterBand(1)
    weights_nodata = weights_band.GetNoDataValue()
    if weights_nodata is None:
        raise ValueError("Weights raster must have a no data value")
    weights_array = weights_band.ReadAsArray().astype(np.float64)

    fac_ds = create_dataset(
        output_fac_path,
        WEIGHTED_ACCUMULATION_NODATA,
        gdal.GDT_Float64,
        fdr_ds.RasterXSize,
        fdr_ds.RasterYSize,
        transform,
        projection,
    )
    fac_band = fac_ds.GetRasterBand(1)
    fac_array, _ = single_tile_flow_accumulation_weighted(
        fdr_array, weights_array, weights_nodata, nodata_mode, False
    )
    fac_band.WriteArray(fac_array)
    fac_band.FlushCache()
    fac_ds.FlushCache()
    fac_band = None
    fac_ds = None
    fdr_ds_band = None
    fdr_ds = None
    weights_band = None
    weights_ds = None


def _flow_accumulation(fdr_path: str, output_fac_path: str) -> None:
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
