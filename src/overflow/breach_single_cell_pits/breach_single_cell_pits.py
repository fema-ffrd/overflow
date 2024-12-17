import numpy as np
from numba import njit, prange
from osgeo import gdal

from overflow.util.raster import raster_chunker, create_dataset
from overflow.util.constants import DEFAULT_CHUNK_SIZE, EPSILON_GRADIENT


@njit(parallel=True)
def breach_single_cell_pits_in_chunk(
    chunk: np.ndarray, nodata_value: float
) -> np.ndarray:
    """
    This function is used to breach single cell pits in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with breached single cell pits.

    Parameters
    ----------
    chunk : np.ndarray
        A chunk of a DEM.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with breached single cell pits.
    """
    dx = [1, 1, 1, 0, -1, -1, -1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]
    dx2 = [2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2, -2, -1, 0, 1]
    dy2 = [-2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2]
    breachcell = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 0]
    # Create a copy of the chunk

    rows, cols = chunk.shape
    # Loop through each cell in the chunk
    unsolved_pits_raster = np.zeros(chunk.shape, dtype=np.int8)
    # pylint: disable=not-an-iterable
    for row in prange(2, rows - 2):
        for col in range(2, cols - 2):
            z = chunk[row, col]
            if z != nodata_value and not np.isnan(z):
                is_flat = True
                is_sink = True
                for k in range(8):
                    zn = chunk[row + dy[k], col + dx[k]]
                    # check if the cell is flat
                    # a cell cannot be flat if one if its neighbors is different
                    if zn != z:
                        is_flat = False
                    # check for sink
                    # a sink is a cell that has no neighbors strictly
                    # lower than it.
                    # nodata is considered the lowest possible value
                    if zn == nodata_value or np.isnan(zn) or zn < z:
                        is_sink = False
                        break

                if not is_flat and is_sink:
                    # Any cell that is a sink and not flat
                    # is marked as an unsolved pit for processing.
                    # This includes cells on the edge of a flat with no lower neighbors,
                    # but with at least one higher neighbor and one equal neighbor
                    # but does not include cells where all nieghbors are equal to the cell
                    # or when a cell as at least one lower neighbor.
                    # By only considering true pits or cells on the edges
                    # of flats this reduces the amount of processing for large flat areas
                    unsolved_pits_raster[row, col] = 1

    pit_indicies = np.argwhere(unsolved_pits_raster == 1)

    for row, col in pit_indicies:
        solved = False
        z = chunk[row, col]
        for k in range(16):
            zn = chunk[row + dy2[k], col + dx2[k]]
            if zn <= z or zn == nodata_value or np.isnan(zn):
                solved = True
                if zn == nodata_value or np.isnan(zn):
                    # we're breaching to a nodata value, so apply a small gradient
                    zn = z - 2 * EPSILON_GRADIENT
                chunk[row + dy[breachcell[k]], col + dx[breachcell[k]]] = (z + zn) / 2
        if solved:
            unsolved_pits_raster[row, col] = 0

    return unsolved_pits_raster


def breach_single_cell_pits(
    input_path: str, output_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE
):
    """
    This function is used to breach single cell pits in a DEM raster.

    Parameters
    ----------
    input_path : str
        The path to the input DEM raster.
    output_path : str
        The path to save the output DEM raster with breached single cell pits.
    chunk_size : int, optional
        The size of the chunks in which the DEM is processed, by default DEFAULT_CHUNK_SIZE.
    """

    input_raster = gdal.Open(input_path)
    projection = input_raster.GetProjection()
    transform = input_raster.GetGeoTransform()

    band = input_raster.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()

    output_ds = create_dataset(
        output_path,
        nodata_value,
        gdal.GDT_Float32,
        input_raster.RasterXSize,
        input_raster.RasterYSize,
        transform,
        projection,
    )
    output_band = output_ds.GetRasterBand(1)

    for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=2):
        _ = breach_single_cell_pits_in_chunk(chunk.data, nodata_value)
        chunk.write(output_band)
