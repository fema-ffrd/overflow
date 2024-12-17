import numpy as np
import pytest
from osgeo import gdal
import click.testing
from overflow.breach_paths_least_cost import (
    breach_all_pits_in_chunk_least_cost,
    breach_paths_least_cost,
    EPSILON_GRADIENT,
    DEFAULT_SEARCH_RADIUS,
    UNVISITED_INDEX,
)
import numba
from overflow_cli import breach_paths_least_cost_cli


@pytest.fixture(name="dem_with_pit")
def fixture_dem_with_pit():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="random_dem")
def fixture_random_dem():
    """A numpy array representing a 1000x1000 DEM with random float values between 1 and 10."""
    np.random.seed(0)  # Ensures reproducibility
    return np.random.uniform(1, 10, size=(1000, 1000)).astype(np.float32)


@pytest.fixture(name="dem_with_pit_and_nodata")
def fixture_dem_with_pit_and_nodata():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [2, 2, 2, -9999, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="dem_from_file")
def fixture_dem_from_file(dem_with_pit):
    """Create a raster filepath for testing."""
    output_path = "/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path, dem_with_pit.shape[0], dem_with_pit.shape[1], 1, gdal.GDT_Float32
    )
    band = dataset.GetRasterBand(1)
    band.WriteArray(dem_with_pit)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


@pytest.fixture(name="random_dem_from_file")
def fixture_random_dem_from_file(random_dem):
    """Create a raster filepath for testing."""
    output_path = "/vsimem/test_raster_breach_random.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path,
        random_dem.shape[0],
        random_dem.shape[1],
        1,
        gdal.GDT_Float32,
    )
    band = dataset.GetRasterBand(1)
    band.WriteArray(random_dem)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_breach_paths_least_cost_chunk_flat():
    """Test that the expected breach path is created for a DEM with a flat area."""
    dem_with_flat = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    pits = np.array([[3, 3]])
    search_radius = DEFAULT_SEARCH_RADIUS
    search_window_size = 2 * search_radius + 1
    chunk_costs_array = np.full(
        (search_window_size, search_window_size),
        np.inf,
        dtype=np.float32,
    )
    chunk_prev_rows_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    chunk_prev_cols_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    breached_dem, _, _ = breach_all_pits_in_chunk_least_cost(
        0,
        0,
        pits,
        dem_with_flat,
        -9999,
        chunk_costs_array,
        chunk_prev_rows_array,
        chunk_prev_cols_array,
        search_radius,
    )
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(breached_dem, expected_dem)


def test_breach_paths_least_cost_chunk(dem_with_pit):
    """Test that the expected breach path is created."""
    pits = np.array([[3, 3]])
    search_radius = DEFAULT_SEARCH_RADIUS
    search_window_size = 2 * search_radius + 1
    chunk_costs_array = np.full(
        (search_window_size, search_window_size),
        np.inf,
        dtype=np.float32,
    )
    chunk_prev_rows_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    chunk_prev_cols_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    breached_dem, _, _ = breach_all_pits_in_chunk_least_cost(
        0,
        0,
        pits,
        dem_with_pit,
        -9999,
        chunk_costs_array,
        chunk_prev_rows_array,
        chunk_prev_cols_array,
        search_radius,
    )
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, -4 / 3, 2, 2, 2],
            [2, 2, 2, -2 / 3, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(breached_dem, expected_dem)


def test_breach_paths_least_cost_chunk_with_nodata(dem_with_pit_and_nodata):
    """Test that the expected breach path is created."""
    pits = np.array([[3, 3]])
    search_radius = DEFAULT_SEARCH_RADIUS
    search_window_size = 2 * search_radius + 1
    chunk_costs_array = np.full(
        (search_window_size, search_window_size),
        np.inf,
        dtype=np.float32,
    )
    chunk_prev_rows_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    chunk_prev_cols_array = np.full(
        (search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    breached_dem, _, _ = breach_all_pits_in_chunk_least_cost(
        0,
        0,
        pits,
        dem_with_pit_and_nodata,
        -9999,
        chunk_costs_array,
        chunk_prev_rows_array,
        chunk_prev_cols_array,
        search_radius,
    )
    expected_dem = np.array(
        [
            [2, 2, 2, -9999, 2, 2, 2],
            [2, 2, 2, -2 * EPSILON_GRADIENT, 2, 2, 2],
            [2, 2, 2, -EPSILON_GRADIENT, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(breached_dem, expected_dem)


def test_dem_from_file(dem_from_file):
    """Same as test_breach_paths_least_cost_chunk but using a file."""
    output_path = "/vsimem/dem_from_file_breached.tif"
    breach_paths_least_cost(
        dem_from_file,
        output_path,
        chunk_size=7,
        search_radius=3,
    )
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, -4 / 3, 2, 2, 2],
            [2, 2, 2, -2 / 3, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(array, expected_dem)
    gdal.Unlink(output_path)


def test_breach_paths_least_cost_random(random_dem_from_file, random_dem):
    """Test that the expected breach path is created."""
    output_path = "/vsimem/test_raster_breach_path_random.tif"
    breach_paths_least_cost(
        random_dem_from_file,
        output_path,
        chunk_size=1000,
        search_radius=100,
    )
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    assert not np.allclose(random_dem, array)
    gdal.Unlink(output_path)


def test_breach_paths_least_cost_cli(dem_from_file):
    """Test the CLI."""
    output_path = "/vsimem/test_breach_paths_least_cost_cli.tif"
    runner = click.testing.CliRunner()
    result = runner.invoke(
        breach_paths_least_cost_cli,
        [
            "--input_file",
            dem_from_file,
            "--output_file",
            output_path,
            "--chunk_size",
            "7",
            "--search_radius",
            "3",
        ],
    )
    assert result.exit_code == 0
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, -4 / 3, 2, 2, 2],
            [2, 2, 2, -2 / 3, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(array, expected_dem)
    gdal.Unlink(output_path)
