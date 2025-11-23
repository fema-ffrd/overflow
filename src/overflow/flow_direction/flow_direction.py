import math

import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.util.constants import (
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTIONS,
    NEIGHBOR_OFFSETS,
)
from overflow.util.progress import ProgressCallback, silent_callback
from overflow.util.raster import create_dataset, raster_chunker


@njit(parallel=True)
def flow_direction_for_tile(dem: np.ndarray, nodata_value: float) -> np.ndarray:
    """
    Define the 8 directions using a list of tuples
       3  |   2    |  1
     ------------------
       4  |   8   |  0
     ------------------
       5  |   6   |  7

    This function is used to calculate flow direction in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with flow direction values.

    Parameters
    ----------
    dem (np.ndarray) : Digital Elevation Model (DEM) chunk.
    nodata_value (float) : Value from dem representing no data

    Returns
    -------
    np.ndarray
        A chunk of a DEM with flow direction values.
    """
    # np.empty is faster than np.full
    # all elements will be set by the algorithm
    fdr = np.empty(dem.shape, dtype=np.uint8)

    # Get the shape of the chunk
    rows, cols = dem.shape

    # Loop through each cell in the chunk

    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):
            if dem[row, col] != nodata_value and not np.isnan(dem[row, col]):
                max_slope = -np.inf
                max_index = -1
                all_non_positive = True

                for i, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
                    slope = calculate_slope(dem, row, col, dy, dx, nodata_value)
                    if slope > max_slope:
                        max_slope = slope
                        max_index = i
                    if slope > 0:
                        all_non_positive = False

                if all_non_positive:
                    fdr[row, col] = FLOW_DIRECTION_UNDEFINED
                else:
                    fdr[row, col] = FLOW_DIRECTIONS[max_index]
            else:
                fdr[row, col] = FLOW_DIRECTION_NODATA

    return fdr


@njit()
def calculate_slope(
    dem: np.ndarray, row: int, col: int, dy: int, dx: int, nodata_value: float
) -> float:
    """
    Calculate the slope between the cell and its neighbors.

    Parameters
    ----------
    dem (np.ndarray) : Digital Elevation Model (DEM).
    row, col (int) : Coordinates of the cell.
    dx, dy (int) : Direction to the neighbor.
    nodata_value (float) : Value representing no data.

    Returns
    -------
    float
        The slope between the cell and its neighbor. Positive slopes indicate downhill flow.
    """
    if dem[row + dy, col + dx] == nodata_value or np.isnan(dem[row + dy, col + dx]):
        return np.inf

    return (dem[row, col] - dem[row + dy, col + dx]) / (  # type: ignore[no-any-return]
        math.sqrt(2) if dx != 0 and dy != 0 else 1
    )


def _flow_direction(
    input_path: str,
    output_path: str,
    chunk_size: int = 4000,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Generates a flow direction raster from a DEM chunks of a given size.

    Parameters
    ----------
    input_path : str
        Path to the input DEM file (GDAL supported raster format).
    output_path : str
        Path to the output flow direction raster file (must be GeoTIFF).
    chunk_size : int, optional
        Size of chunks to process at a time, by default 4000.
    progress_callback : ProgressCallback | None, optional
        Optional callback for progress reporting, by default None (silent).

    Returns
    -------
    None
    """
    if progress_callback is None:
        progress_callback = silent_callback

    input_raster = gdal.Open(input_path)
    projection = input_raster.GetProjection()
    transform = input_raster.GetGeoTransform()

    band = input_raster.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()

    output_ds = create_dataset(
        output_path,
        FLOW_DIRECTION_NODATA,
        gdal.GDT_Byte,
        input_raster.RasterXSize,
        input_raster.RasterYSize,
        transform,
        projection,
    )
    output_band = output_ds.GetRasterBand(1)

    # Calculate direction
    progress_callback(
        step_name="Calculate direction", step_number=1, total_steps=1, progress=0.0
    )

    for chunk in raster_chunker(
        band,
        chunk_size=chunk_size,
        chunk_buffer_size=1,
        progress_callback=progress_callback,
    ):
        result = flow_direction_for_tile(chunk.data, nodata_value)
        chunk.from_numpy(result)
        chunk.write(output_band)

    # Explicitly flush cache and close datasets to ensure
    # data is fully written to disk before next step reads it
    output_band.FlushCache()
    output_ds.FlushCache()
    output_band = None
    output_ds = None
    band = None
    input_raster = None
