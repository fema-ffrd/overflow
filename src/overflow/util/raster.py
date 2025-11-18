import sys
from collections.abc import Generator
from enum import Enum
from typing import Any

import numpy as np
from numba import float32, int64, njit  # type: ignore[attr-defined]
from numba.experimental import jitclass
from osgeo import gdal, osr
from rich.console import Console

from overflow.util.constants import NEIGHBOR_OFFSETS

gdal.UseExceptions()


def gdal_data_type_to_numpy_data_type(gdal_dtype: int) -> np.dtype:
    """Map a GDAL data type to a numpy data type.

    Args:
        gdal_dtype (int): The GDAL data type to map.

    Returns:
        np.dtype: The numpy data type that corresponds to the input GDAL data type.
    """
    # map GDAL data type to numpy data type
    gdal_numpy_dtype_mapping = {
        "Byte": np.uint8,
        "UInt16": np.uint16,
        "Int16": np.int16,
        "UInt32": np.uint32,
        "Int32": np.int32,
        "UInt64": np.uint64,
        "Int64": np.int64,
        "Float32": np.float32,
        "Float64": np.float64,
        "CInt16": np.complex64,
        "CInt32": np.complex128,
        "CFloat32": np.complex64,
        "CFloat64": np.complex128,
    }
    return gdal_numpy_dtype_mapping[gdal.GetDataTypeName(gdal_dtype)]  # type: ignore[return-value]


def read_raster_with_bounds_handling(
    x_offset: int, y_offset: int, x_size: int, y_size: int, raster_band: gdal.Band
) -> np.ndarray:
    """Read a chunk of a raster band and return it as a numpy array. This function allows for reading
       out of bounds regions. If the window, or part of the window, extends beyond the edge of the raster,
       the out of bounds region will be filled with the nodata value for the band.

    Args:
        x_offset (int): The x offset of the chunk to read.
        y_offset (int): The y offset of the chunk to read.
        x_size (int): The number of columns in the chunk.
        y_size (int): The number of rows in the chunk.
        raster_band (gdal.Band): The raster band to read from.

    Returns:
        np.ndarray: The chunk of the raster band as a numpy array. The shape of the array will be: (y_size, x_size).
    """
    assert x_size >= 0, "x_size must be positive"
    assert y_size >= 0, "y_size must be positive"
    # Get the no data value from the raster band
    no_data_value = raster_band.GetNoDataValue()
    assert no_data_value is not None, "The raster band has no no data value"
    # Get the GDAL data type from the raster band
    gdal_dtype = raster_band.DataType

    # Convert the GDAL data type to a numpy data type
    np_dtype = gdal_data_type_to_numpy_data_type(gdal_dtype)

    # Create an array filled with the no data value
    window_data = np.full((y_size, x_size), no_data_value, dtype=np_dtype)

    # Calculate the offsets and sizes for reading the array
    # x_offset_adjusted and y_offset_adjusted are ensuring that the reading process doesn't start before the
    # beginning of the band. If x_offset or y_offset are negative, they are set to 0.
    x_offset_adjusted = max(x_offset, 0)
    y_offset_adjusted = max(y_offset, 0)

    # if x or y offsets are adjusted, the size of the window needs to be adjusted
    # by the same amount. This is to ensure that the returned window size remains the same
    # as the requested window size.
    x_size_adjusted = x_size - (x_offset_adjusted - x_offset)
    y_size_adjusted = y_size - (y_offset_adjusted - y_offset)

    # Calculate how much of the band remains after the offset.
    # If the offset is beyond the end of the band, this will be 0.
    x_remaining = max(raster_band.XSize - x_offset_adjusted, 0)
    y_remaining = max(raster_band.YSize - y_offset_adjusted, 0)

    # win_xsize and win_ysize are calculating the size of the window to read. They are ensuring that the
    # window doesn't extend beyond the end of the band by choosing the smaller of given y_size_adjusted
    # and y_size_adjusted and the size remaining.
    win_xsize = min(x_size_adjusted, x_remaining)
    win_ysize = min(y_size_adjusted, y_remaining)

    # Write band data to window data at the requested offsets
    window_data[
        y_offset_adjusted - y_offset : y_offset_adjusted - y_offset + win_ysize,
        x_offset_adjusted - x_offset : x_offset_adjusted - x_offset + win_xsize,
    ] = raster_band.ReadAsArray(
        xoff=x_offset_adjusted,
        yoff=y_offset_adjusted,
        win_xsize=win_xsize,
        win_ysize=win_ysize,
    )

    return window_data


class RasterChunk:
    """A class to represent a chunk of a raster band including an overlapping buffer region on all edges."""

    def __init__(
        self,
        row: int,
        col: int,
        size: int,
        buffer_size: int,
    ):
        self.data: np.ndarray | None = None
        self.row = row
        self.col = col
        self.size = size
        self.buffer_size = buffer_size

    def from_numpy(self, data: np.ndarray):
        """Create a RasterChunk object from a numpy array.

        Args:
            data (np.ndarray): The numpy array to create the RasterChunk object from.
        """
        self.data = data

    def read(self, band: gdal.Band):
        """Read a chunk of a raster band including an overlapping buffer region on all edges.
           If part of the chunk, including the buffer region, extends beyond the edge of the raster,
           the out of bounds region will be filled with nodata. The chunk will be stored as a numpy array
           in the data attribute of the RasterChunk object.

        Args:
            band (gdal.Band): The raster band to read from.
        """
        x_offset = self.col * self.size - self.buffer_size
        y_offset = self.row * self.size - self.buffer_size
        x_size = self.size + 2 * self.buffer_size
        y_size = self.size + 2 * self.buffer_size
        self.data = read_raster_with_bounds_handling(
            x_offset, y_offset, x_size, y_size, band
        )

    def _get_unbuffered_data(self, band: gdal.Band) -> np.ndarray:
        assert self.data is not None
        y_remaining = max(band.YSize - self.row * self.size, 0)
        x_remaining = max(band.XSize - self.col * self.size, 0)
        unbuffered_y_size = min(self.size, y_remaining)
        unbuffered_x_size = min(self.size, x_remaining)
        return self.data[
            self.buffer_size : self.buffer_size + unbuffered_y_size,
            self.buffer_size : self.buffer_size + unbuffered_x_size,
        ]

    def write(self, band: gdal.Band):
        """Write a chunk to a raster band. The chunk must have been read from a band of the same size.
        The chunk will have only it's unbuffered data written to the band and will not write out of bounds regions.

        Args:
            band (gdal.Band): The raster band to write to.
        """
        if self.data is not None:
            band.WriteArray(
                self._get_unbuffered_data(band),
                xoff=self.col * self.size,
                yoff=self.row * self.size,
            )
        else:
            raise ValueError("The chunk has not been read yet.")


def raster_chunker(
    band: gdal.Band, chunk_size: int, chunk_buffer_size: int = 0, lock=None
) -> Generator[RasterChunk, Any, None]:
    """Generator that yields chunks of a raster.

    Args:
        band  (gdal.Band): The raster band to read from.
        chunk_row_size (int): The number of rows in each chunk.
        chunk_col_size (int): The number of columns in each chunk.
        buffer_row_size (int): The number of rows in the buffer region.
        buffer_col_size (int): The number of columns in the buffer region.

    Yields:
        Generator[RasterChunk]: A generator that yields a RasterChunk.
    """
    # Calculate the number of chunks in each dimension
    n_chunks_row = (band.YSize + chunk_size - 1) // chunk_size
    n_chunks_col = (band.XSize + chunk_size - 1) // chunk_size
    # Calculate the total number of chunks
    total_chunks = n_chunks_row * n_chunks_col
    console = Console()
    is_a_tty = sys.stdout.isatty()
    # Iterate over the chunks
    with console.status("[bold green]Processing Chunks: ") as status:
        for chunk_row in range(n_chunks_row):
            for chunk_col in range(n_chunks_col):
                # Read the chunk and yield it
                chunk = RasterChunk(
                    chunk_row,
                    chunk_col,
                    chunk_size,
                    chunk_buffer_size,
                )
                if lock:
                    with lock:
                        chunk.read(band)
                else:
                    chunk.read(band)
                yield chunk
                # Log progress
                if is_a_tty:
                    status.update(
                        f"[bold green]Processing Chunks: {chunk_row * n_chunks_col + chunk_col + 1}/{total_chunks}"
                    )
                else:
                    print(
                        f"Processing Chunks: {chunk_row * n_chunks_col + chunk_col + 1}/{total_chunks}",
                        end="\r",
                        flush=True,
                    )


def create_grid_cell_class(value_type):
    @jitclass
    class GridCell:
        """A class to represent a cell in the grid. Used with heapq to prioritize cells by cost."""

        row: int64  # type: ignore[annotation-unchecked]
        col: int64  # type: ignore[annotation-unchecked]
        value: value_type  # type: ignore[annotation-unchecked]

        def __init__(self, row, col, value):
            self.row = row
            self.col = col
            self.value = value

        # Define comparison methods based on the cost attribute so this can be used in a heapq
        def __lt__(self, other):
            return self.value < other.value

        def __eq__(self, other):
            return self.value == other.value

        def __le__(self, other):
            return self.value <= other.value

        def __gt__(self, other):
            return self.value > other.value

        def __ge__(self, other):
            return self.value >= other.value

        def __ne__(self, other):
            return self.value != other.value

    return GridCell


GridCellInt64 = create_grid_cell_class(int64)
GridCellFloat32 = create_grid_cell_class(float32)


@jitclass
class GridCoordinate:
    row: int64
    col: int64

    def __init__(self, row, col):
        self.row = row
        self.col = col


class Side(Enum):
    """
    An enumeration representing the sides of a 2D array (tile).
    Used to join tiles together along their edges.
    """

    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


class Corner(Enum):
    """
    An enumeration representing the corners of a 2D array (tile).
    Used to join tiles together at their corners.
    """

    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4


@njit
def get_tile_perimeter(array: np.ndarray) -> np.ndarray:
    """
    Get the perimeter of a 2D array (tile) as a 1D array.
    The perimeter is ordered in a clockwise direction starting from the top-left corner.
    """
    rows = len(array)
    cols = len(array[0])
    perimeter = np.empty(2 * (rows + cols) - 4, dtype=array.dtype)
    # Fill the perimeter array
    perimeter[:cols] = array[0]  # top
    perimeter[cols : cols + rows - 2] = array[1:-1, -1]  # right
    perimeter[cols + rows - 2 : 2 * cols + rows - 2] = array[-1, ::-1]  # bottom
    perimeter[2 * cols + rows - 2 :] = array[-2:0:-1, 0]  # left
    return perimeter


@njit
def neighbor_generator(row: int, col: int, n_row: int, n_col: int):
    """
    A generator function that yields the coordinates of the neighbors of a given cell in a 2D grid.

    The function takes as input the coordinates of a cell (row, col) and the dimensions of the grid (n_row, n_col).
    It generates the coordinates of the 8 neighboring cells (top, bottom, left, right, and the 4 diagonals)
    if they are within the grid boundaries.

    Parameters:
    row (int): The row index of the cell.
    col (int): The column index of the cell.
    n_row (int): The total number of rows in the grid.
    n_col (int): The total number of columns in the grid.

    Yields:
    tuple: A tuple containing the row and column indices of a neighboring cell.

    Example:
    >>> list(neighbor_generator(1, 1, 3, 3))
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    """
    for d_row, d_col in NEIGHBOR_OFFSETS:
        neighbor_row = row + d_row
        neighbor_col = col + d_col
        not_in_bounds = (
            neighbor_row < 0
            or neighbor_row >= n_row
            or neighbor_col < 0
            or neighbor_col >= n_col
        )
        if not_in_bounds:
            continue
        yield neighbor_row, neighbor_col


def open_dataset(filepath: str, access: int = gdal.GA_ReadOnly) -> gdal.Dataset:
    """
    Opens a raster dataset and returns the dataset and specified band.

    Args:
        filepath (str): The path to the raster file.
        band_number (int, optional): The band number to open (default is 1).
        access (int, optional): The access mode for opening the dataset (default is gdal.GA_ReadOnly).

    Returns:
        gdal.Dataset: the opened dataset (gdal.Dataset)

    Raises:
        ValueError: If unable to open the dataset.
    """
    dataset = gdal.Open(filepath, access)
    if dataset is None:
        raise ValueError(f"Unable to open dataset: {filepath}")
    return dataset


def create_dataset(
    filepath: str,
    nodata_value: float | int,
    data_type: int,
    x_size: int,
    y_size: int,
    geotransform: tuple | None = None,
    projection: str | None = None,
) -> gdal.Dataset:
    """
    Creates a new raster dataset with the specified parameters using ZSTD compression.

    Args:
        filepath (str): The path to save the created dataset.
        nodata_value (float|int): The value to use for representing nodata pixels.
        data_type (int): The data type of the raster dataset (e.g., gdal.GDT_Float32).
        x_size (int): The width of the raster dataset in pixels.
        y_size (int): The height of the raster dataset in pixels.
        geotransform (tuple, optional): The geotransform parameters for the dataset (default is None).
        projection (str, optional): The projection information for the dataset (default is None).
        num_bands (int, optional): The number of bands in the dataset (default is 1).

    Returns:
        gdal.Dataset: the created dataset
    """
    try:
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            filepath,
            x_size,
            y_size,
            1,
            data_type,
            options=[
                "COMPRESS=ZSTD",
                "TILED=YES",
                "BIGTIFF=YES",
                "NUM_THREADS=ALL_CPUS",
            ],
        )
        if geotransform:
            dataset.SetGeoTransform(geotransform)
        if projection:
            dataset.SetProjection(projection)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(nodata_value)
        return dataset
    except Exception as e:
        raise ValueError(f"Error creating dataset '{filepath}': {e}") from e


def read_tile(filepath: str) -> np.ndarray:
    """
    Reads a raster tile from the specified file path and returns its data as a numpy array.

    Args:
        filepath (str): The path to the raster tile file.

    Returns:
        numpy.ndarray: The raster tile data as a numpy array.
    """
    ds = open_dataset(filepath)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()  # type: ignore[no-any-return]


def mosaic_tiles(
    prefix: str,
    rows: int,
    cols: int,
    output_filepath: str,
    nodata_value: float | int,
    data_type: int,
    x_size: int,
    y_size: int,
    geotransform: tuple,
    projection: str,
) -> None:
    """
    Mosaics multiple raster tiles into a single raster dataset.

    Args:
        prefix (str): The prefix of the tile filepaths. '_{row}_{col}.tif' will be appended to the prefix
            for every row, col in given number of rows and columns.
        rows (int): The number of rows in the tile grid.
        cols (int): The number of columns in the tile grid.
        output_filepath (str): The path to save the mosaicked raster dataset.
        nodata_value (float|int): The value to use for representing nodata pixels.
        data_type (int): The data type of the output raster dataset (e.g., gdal.GDT_Float32).
        x_size (int): The width of the output raster dataset in pixels.
        y_size (int): The height of the output raster dataset in pixels.
        geotransform (tuple): The geotransform parameters for the output dataset.
        projection (str): The projection information for the output dataset.
    """
    out_ds = create_dataset(
        output_filepath,
        nodata_value,
        data_type,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    out_band = out_ds.GetRasterBand(1)
    for row in range(rows):
        for col in range(cols):
            tile_filepath = f"{prefix}_{row}_{col}.tif"
            tile = read_tile(tile_filepath)
            chunk_size = tile.shape[0]
            chunk = RasterChunk(row, col, chunk_size, 0)
            chunk.from_numpy(tile)
            chunk.write(out_band)
    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    out_band = None


class TileManager:
    def __init__(self, raster_path, tile_size, max_cache_size):
        self.raster_path = raster_path
        self.tile_size = tile_size
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_order = []
        self.dirty_tiles = set()

        self.dataset = gdal.Open(raster_path, gdal.GA_Update)
        self.raster_width = self.dataset.RasterXSize
        self.raster_height = self.dataset.RasterYSize
        self.band = self.dataset.GetRasterBand(1)

        self.tile_cols = (self.raster_width + tile_size - 1) // tile_size
        self.tile_rows = (self.raster_height + tile_size - 1) // tile_size

    def __del__(self):
        self.flush_cache()
        self.dataset = None

    def get_tile(self, tile_col, tile_row):
        tile_key = (tile_col, tile_row)
        if tile_key in self.cache:
            self.cache_order.remove(tile_key)
            self.cache_order.append(tile_key)
            return self.cache[tile_key]
        else:
            tile_data = self.load_tile(tile_col, tile_row)
            self.cache[tile_key] = tile_data
            self.cache_order.append(tile_key)
            if len(self.cache) > self.max_cache_size:
                evicted_tile_key = self.cache_order.pop(0)
                self.flush_tile(evicted_tile_key)
                del self.cache[evicted_tile_key]
            return tile_data

    def load_tile(self, tile_col, tile_row):
        offset_x = tile_col * self.tile_size
        offset_y = tile_row * self.tile_size
        tile_width = min(self.tile_size, self.raster_width - offset_x)
        tile_height = min(self.tile_size, self.raster_height - offset_y)

        tile_data = self.band.ReadAsArray(offset_x, offset_y, tile_width, tile_height)
        return tile_data

    def flush_tile(self, tile_key):
        if tile_key in self.dirty_tiles:
            tile_col, tile_row = tile_key
            tile_data = self.cache[tile_key]
            offset_x = tile_col * self.tile_size
            offset_y = tile_row * self.tile_size
            min(self.tile_size, self.raster_width - offset_x)
            min(self.tile_size, self.raster_height - offset_y)

            self.band.WriteArray(tile_data, offset_x, offset_y)
            self.dirty_tiles.remove(tile_key)

    def flush_cache(self):
        for tile_key in self.cache:
            self.flush_tile(tile_key)
        self.cache.clear()
        self.cache_order.clear()
        self.dirty_tiles.clear()

    def get_cell_value(self, x, y):
        tile_col = x // self.tile_size
        tile_row = y // self.tile_size
        tile_data = self.get_tile(tile_col, tile_row)

        tile_x = x % self.tile_size
        tile_y = y % self.tile_size

        return tile_data[tile_y, tile_x]

    def set_cell_value(self, x, y, value):
        tile_col = x // self.tile_size
        tile_row = y // self.tile_size
        tile_key = (tile_col, tile_row)
        tile_data = self.get_tile(tile_col, tile_row)

        tile_x = x % self.tile_size
        tile_y = y % self.tile_size

        tile_data[tile_y, tile_x] = value
        self.dirty_tiles.add(tile_key)


@njit
def cell_to_coords(
    i: np.int32,
    j: np.int32,
    gt: np.ndarray,
    tile_row: np.int32 = 0,  # type: ignore[assignment]
    tile_col: np.int32 = 0,  # type: ignore[assignment]
    chunk_size: np.int32 = 0,  # type: ignore[assignment]
) -> tuple[np.float64, np.float64]:
    """
    Convert raster cell indices to geographic coordinates using numerically stable calculations.

    Args:
        i (int): Row index (0 is top row)
        j (int): Column index (0 is leftmost column)
        gt (ndarray): GDAL geotransform
        tile_row (int): Row index of the current tile
        tile_col (int): Column index of the current tile
        chunk_size (int): Size of each tile
    Returns:
        tuple: (x, y) coordinates in the raster's projection as float64
    """
    # Cast to int64 for intermediate calculations to prevent overflow
    global_i = np.float64(np.int64(tile_row) * np.int64(chunk_size) + np.int64(i))
    global_j = np.float64(np.int64(tile_col) * np.int64(chunk_size) + np.int64(j))

    # Calculate coordinates
    x = np.float64(gt[0])  # Origin x
    x += (global_j + 0.5) * gt[1]  # Pixel width contribution
    x += global_i * gt[2]  # Row rotation contribution

    y = np.float64(gt[3])  # Origin y
    y += global_j * gt[4]  # Column rotation contribution
    y += (global_i + 0.5) * gt[5]  # Pixel height contribution

    return np.float64(x), np.float64(y)


@njit
def coords_to_cell(
    x: np.float64,
    y: np.float64,
    gt: np.ndarray,
    tile_row: np.int32 = 0,  # type: ignore[assignment]
    tile_col: np.int32 = 0,  # type: ignore[assignment]
    chunk_size: np.int32 = 0,  # type: ignore[assignment]
) -> tuple[np.int32, np.int32]:
    """
    Convert geographic coordinates to raster cell indices using numerically stable calculations.
    This is the inverse of cell_to_coords.

    Args:
        x (float64): X coordinate in the raster's projection
        y (float64): Y coordinate in the raster's projection
        gt (ndarray): GDAL geotransform
        tile_row (int): Row index of the current tile
        tile_col (int): Column index of the current tile
        chunk_size (int): Size of each tile
    Returns:
        tuple: (i, j) cell indices where i is row (0 is top) and j is column (0 is leftmost)
    """
    # Convert input coordinates to float64 for numerical stability
    x = np.float64(x)
    y = np.float64(y)

    # Calculate determinant for inverse transformation
    det = np.float64(gt[1] * gt[5] - gt[2] * gt[4])
    if abs(det) < 1e-10:  # Check for singular matrix
        raise ValueError("Geotransform matrix is singular")

    # Calculate relative coordinates from origin
    dx = x - np.float64(gt[0])
    dy = y - np.float64(gt[3])

    # Solve the system of equations using matrix inverse
    # [gt[1] gt[2]] [j] = [dx]
    # [gt[4] gt[5]] [i] = [dy]

    # Calculate global indices using inverse matrix
    global_j = np.float64(gt[5] * dx - gt[2] * dy) / det
    global_i = np.float64(-gt[4] * dx + gt[1] * dy) / det

    # Subtract 0.5 to reverse the pixel center offset from cell_to_coords
    global_j -= 0.5
    global_i -= 0.5

    # Convert to local tile coordinates
    tile_offset_i = np.int64(tile_row) * np.int64(chunk_size)
    tile_offset_j = np.int64(tile_col) * np.int64(chunk_size)

    # Calculate local indices
    i = np.int32(round(global_i - tile_offset_i))
    j = np.int32(round(global_j - tile_offset_j))

    return i, j


@njit
def grid_hash(i: int, j: int) -> int:
    """
    Create a hash from grid indices, supporting negative indices.

    This function implements the Szudzik pairing function to create a unique
    hash for each pair of integers, including negative values.

    Args:
        i (int): Grid row index
        j (int): Grid column index

    Returns:
        int: Unique hash for the grid cell
    """
    a = np.uint64(2 * abs(i) if i >= 0 else 2 * abs(i) - 1)
    b = np.uint64(2 * abs(j) if j >= 0 else 2 * abs(j) - 1)
    c = np.int64((a * a + a + b if a >= b else a + b * b) // 2)
    return c if (i < 0 and j < 0) or (i >= 0 and j >= 0) else -c - 1  # type: ignore[return-value]


def get_units_to_meters_conversion(raster_ds: gdal.Dataset) -> float:
    """
    Get the conversion factor from a raster's spatial reference units to meters.

    Parameters:
    -----------
    raster_ds : gdal.Dataset
        Dataset of the raster.

    Returns:
    --------
    float
        Conversion factor to multiply by to convert from the raster's units to meters.
        Returns 1.0 if units are already in meters or if units cannot be determined.

    Examples:
    --------
    # For a raster in degrees (WGS84)
    >>> factor = get_units_to_meters_conversion('wgs84_raster.tif')
    # factor will be approximately 111319.49079327357 at the equator

    # For a raster in feet
    >>> factor = get_units_to_meters_conversion('nad83_feet_raster.tif')
    # factor will be 0.3048
    """

    # Open the raster
    ds = raster_ds
    if ds is None:
        raise ValueError("Unable to open the raster dataset")

    # Get the spatial reference
    proj = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    # Get the linear units
    linear_units = srs.GetLinearUnits()

    # Check if we're dealing with a geographic coordinate system
    if srs.IsGeographic():
        # For geographic coordinates, we need to calculate the meters per degree
        # This varies with latitude, so we'll use the center of the raster

        # Get raster dimensions
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()

        # Calculate center coordinates
        center_x = gt[0] + width / 2 * gt[1]
        center_y = gt[3] + height / 2 * gt[5]

        # Create a point slightly offset in longitude at the same latitude
        long_offset = 1.0  # one degree

        # Create transformation to meters (EPSG:3857 - Web Mercator)
        target = osr.SpatialReference()
        target.ImportFromEPSG(3857)
        transform = osr.CoordinateTransformation(srs, target)

        # Convert both points and calculate distance
        x1, y1, _ = transform.TransformPoint(center_x, center_y)
        x2, y2, _ = transform.TransformPoint(center_x + long_offset, center_y)

        # Calculate distance in meters per degree
        meters_per_degree = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / long_offset

        return meters_per_degree  # type: ignore[no-any-return]
    else:
        # For projected coordinates, return the linear unit conversion factor
        return linear_units  # type: ignore[no-any-return]


def feet_to_cell_count(feet: float, raster_path: str) -> int:
    """
    Convert feet to cell count based on the cell size of a raster.

    Args:
        feet (float): Feet.
        raster_path (str): Path to the raster file.

    Returns:
        int: Number of cells equivalent to the given feet.
    """
    dataset = gdal.Open(raster_path)
    units_to_meters = get_units_to_meters_conversion(dataset)
    # convert feet to meters
    meters = feet * 0.3048
    # get the cell size in meters
    cell_size_x = dataset.GetGeoTransform()[1] * units_to_meters
    # convert meters to cell count
    cell_count = meters / cell_size_x
    return int(cell_count)


SQ_MILES_TO_SQ_METERS = 2589988.11


def sqmi_to_cell_count(sqmi: float, raster_path: str) -> int:
    """
    Convert square miles to cell count based on the area of a raster.

    Args:
        sqmi (float): Square miles.
        raster_path (str): Path to the raster file.

    Returns:
        int: Number of cells equivalent to the given square miles.
    """
    dataset = gdal.Open(raster_path)
    units_to_meters = get_units_to_meters_conversion(dataset)
    # get the cell size in meters
    cell_size_x = dataset.GetGeoTransform()[1] * units_to_meters
    cell_size_y = -dataset.GetGeoTransform()[5] * units_to_meters
    # convert square miles to square meters
    sqm = sqmi * SQ_MILES_TO_SQ_METERS
    # convert square meters to cell count
    cell_count = sqm / (cell_size_x * cell_size_y)
    return int(cell_count)
