import numpy as np
import pytest
from osgeo import gdal

from overflow._breach_single_cell_pits import (
    breach_single_cell_pits,
    breach_single_cell_pits_in_chunk,
)

# Note: breach_single_cell_pits CLI has been removed in favor of the unified breach command


@pytest.fixture(name="raster_file_path")
def fixture_raster_file_path():
    """Create a random raster band for testing.


    Yields:
        gdal.Band: A raster band of size 5x5 with random float32 data.
    """
    output_path = "/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.array(
        [
            [2, 2, 2, 2, 2],
            [-1, 2, 2, 2, 2],
            [2, 2, 0, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    band.WriteArray(array)
    band.SetNoDataValue(-np.inf)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


@pytest.fixture(name="dem_chunk")
def fixture_dem_chunk():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    return np.array(
        [
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, -1, 2, 2, 2, 2, -999, -999],
            [-999, -999, 2, 2, 0, 2, 2, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
        ],
        dtype=np.float32,
    )


def test_breach_cingle_cell_pits_in_chunk(dem_chunk):
    """Test the single cell pits are breached in a chunk."""
    expected = np.array(
        [
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, -1, 2, 2, 2, 2, -999, -999],
            [-999, -999, 2, -0.5, 0, 2, 2, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, 2, 2, 2, 2, 2, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
        ],
        dtype=np.float32,
    )
    nodata_value = -999
    breach_single_cell_pits_in_chunk(dem_chunk, nodata_value)
    assert np.allclose(dem_chunk, expected)


def test_breach_single_cell_pits(raster_file_path):
    """Test the single cell pits are breached in a raster file."""
    expected = np.array(
        [
            [2, 2, 2, 2, 2],
            [-1, 2, 2, 2, 2],
            [2, -0.5, 0, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    results_path = "/vsimem/test_breach_single_cell_pits.tif"
    breach_single_cell_pits(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    band_array = band.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize)
    assert np.allclose(band_array, expected)

    # Note: CLI test removed - breach_single_cell_pits CLI has been removed
    # in favor of the unified 'breach' command which uses breach_paths_least_cost
