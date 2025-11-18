import math

import numpy as np
import pytest
from osgeo import gdal

from overflow.util.raster import raster_chunker, read_raster_with_bounds_handling

band_fixtures = ["square_raster_band", "tall_raster_band", "wide_raster_band"]


@pytest.fixture(name="raster_band", params=band_fixtures, scope="module")
def fixture_raster_band(request):
    """Create a random raster band for testing. Parametrized to test all band sizes."""
    return request.getfixturevalue(request.param)


@pytest.fixture(name="square_raster_band", scope="module")
def fixture_square_raster_band():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", 100, 100, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.random.rand(100, 100).astype(np.float32)
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    yield band


@pytest.fixture(name="tall_raster_band", scope="module")
def fixture_tall_raster_band():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x200 with random float32 data.
    """
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", 100, 200, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.random.rand(100, 100).astype(np.float32)
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    yield band


@pytest.fixture(name="wide_raster_band", scope="module")
def fixture_wide_raster_band():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 200x100 with random float32 data.
    """
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", 200, 100, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.random.rand(100, 100).astype(np.float32)
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    yield band


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_in_bounds(raster_band: gdal.Band):
    """Test reading a raster in bounds"""
    array = read_raster_with_bounds_handling(10, 10, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.ReadAsArray(10, 10, 20, 20))


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_out_of_bounds(raster_band):
    """Test reading a raster out of bounds"""
    array = read_raster_with_bounds_handling(90, 90, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array[10:, 10:] == raster_band.GetNoDataValue())


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_negative_offset(raster_band):
    """Test reading a raster with a negative offset"""
    array = read_raster_with_bounds_handling(-10, -10, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array[:10, :10] == raster_band.GetNoDataValue())


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_offset_larger_than_raster(raster_band):
    """Test reading a raster with an offset larger than the raster"""
    array = read_raster_with_bounds_handling(100, 100, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.GetNoDataValue())
    array = read_raster_with_bounds_handling(200, 200, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.GetNoDataValue())


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_zero_size(raster_band):
    """Test reading a raster with a zero size"""
    array = read_raster_with_bounds_handling(10, 10, 0, 20, raster_band)
    assert array.shape == (20, 0)
    array = read_raster_with_bounds_handling(10, 10, 20, 0, raster_band)
    assert array.shape == (0, 20)
    array = read_raster_with_bounds_handling(10, 10, 0, 0, raster_band)
    assert array.shape == (0, 0)


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_read_raster_as_array_negative_size(raster_band):
    """Test reading a raster with a negative size"""
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, -20, 20, raster_band)
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, 20, -20, raster_band)
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, -20, -20, raster_band)


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_raster_chunker_even(raster_band):
    """Test the raster chunker with a chunk size that evenly divides the raster size."""
    chunk_size = 10
    buffer_size = 2
    last_row = math.ceil(raster_band.YSize / chunk_size) - 1
    last_col = math.ceil(raster_band.XSize / chunk_size) - 1

    # create temp in memory raster band the same size as the original
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create(
        "test_raster_chunker_even",
        raster_band.XSize,
        raster_band.YSize,
        1,
        gdal.GDT_Float32,
    )
    temp_band = dataset.GetRasterBand(1)

    for chunk in raster_chunker(raster_band, chunk_size, buffer_size):
        # Check the size of the chunk
        assert chunk.data.shape == (
            chunk_size + 2 * buffer_size,
            chunk_size + 2 * buffer_size,
        )
        # check the size of the unbuffered data
        assert chunk._get_unbuffered_data(raster_band).shape == (chunk_size, chunk_size)
        # for the first row of chunks, there should be nodata in the top buffer region
        if chunk.row == 0:
            assert np.all(chunk.data[:buffer_size, :] == raster_band.GetNoDataValue())
        # for the first column of chunks, there should be nodata in the left buffer region
        if chunk.col == 0:
            assert np.all(chunk.data[:, :buffer_size] == raster_band.GetNoDataValue())
        # for the last row of chunks, there should be nodata in the bottom buffer region
        if chunk.row == last_row:
            assert np.all(chunk.data[-buffer_size:, :] == raster_band.GetNoDataValue())
        # for the last column of chunks, there should be nodata in the right buffer region
        if chunk.col == last_col:
            assert np.all(chunk.data[:, -buffer_size:] == raster_band.GetNoDataValue())
        # for the inner chunks, the data should match the original raster
        if 0 < chunk.row < last_row and 0 < chunk.col < last_col:
            assert np.all(
                chunk.data
                == raster_band.ReadAsArray(
                    chunk.col * chunk_size - buffer_size,
                    chunk.row * chunk_size - buffer_size,
                    chunk_size + 2 * buffer_size,
                    chunk_size + 2 * buffer_size,
                )
            )
        # write the chunk to the temp band
        chunk.write(temp_band)
    # flush the temp band to make sure the data is written
    temp_band.FlushCache()
    # read the temp band and check that it matches the original raster
    assert np.all(temp_band.ReadAsArray() == raster_band.ReadAsArray())


@pytest.mark.parametrize(
    "raster_band",
    ["square_raster_band", "tall_raster_band", "wide_raster_band"],
    indirect=True,
)
def test_raster_chunker_odd(raster_band):
    """Test the raster chunker with a chunk size that does not evenly divide the raster size."""
    chunk_size = 11
    buffer_size = 2
    last_row = math.ceil(raster_band.YSize / chunk_size) - 1
    last_col = math.ceil(raster_band.XSize / chunk_size) - 1
    row_remainder = raster_band.YSize % chunk_size
    col_remainder = raster_band.XSize % chunk_size

    # create temp in memory raster band the same size as the original
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create(
        "test_raster_chunker_odd",
        raster_band.XSize,
        raster_band.YSize,
        1,
        gdal.GDT_Float32,
    )
    temp_band = dataset.GetRasterBand(1)

    for chunk in raster_chunker(raster_band, chunk_size, buffer_size):
        # Check the size of the chunk
        assert chunk.data.shape == (
            chunk_size + 2 * buffer_size,
            chunk_size + 2 * buffer_size,
        )
        if chunk.row < last_row and chunk.col < last_col:
            # unbuffered size is always the chunk size in this case
            assert chunk._get_unbuffered_data(raster_band).shape == (
                chunk_size,
                chunk_size,
            )
        else:
            # unbuffered size is smaller than the chunk size in this case
            ub_shape = chunk._get_unbuffered_data(raster_band).shape
            if chunk.col == last_col and chunk.row < last_row:
                assert ub_shape == (chunk_size, col_remainder)
            elif chunk.row == last_row and chunk.col < last_col:
                assert ub_shape == (row_remainder, chunk_size)
            else:
                assert ub_shape == (row_remainder, col_remainder)
        # write the chunk to the temp band
        chunk.write(temp_band)
    # flush the temp band to make sure the data is written
    temp_band.FlushCache()
    # read the temp band and check that it matches the original raster
    assert np.all(temp_band.ReadAsArray() == raster_band.ReadAsArray())
