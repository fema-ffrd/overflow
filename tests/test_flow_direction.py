import click.testing
import numpy as np
import pytest
from osgeo import gdal

from overflow._flow_direction import _flow_direction, flow_direction_for_tile
from overflow._util.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_WEST,
)
from overflow.cli import flow_direction_cli


@pytest.fixture(name="raster_file_path")
def fixture_raster_file_path():
    """Create a raster band for testing.

    Yields:
        gdal.Band: A raster band of size 5x5.
    """
    output_path = "/vsimem/test_raster_FDIR.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    # row 1 - all cells should flow to the right, final cell is undefined
    # row 2 - all cells should flow north east except final 2 cells drain north
    # all cells bordering 4 should drain into 4
    array = np.array(
        [
            [5, 4, 3, 2, 1],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 4, 5, 5],
            [5, 5, 5, 5, 5],
        ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


@pytest.fixture(name="dem")
def fixture_dem():
    """Create a dem for testing.

    Returns:
        np.ndarray: A dem of size 5x5.
    """
    return np.array(
        [
            [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
            [-9999, 5, 4, 3, 2, 1, -9999],
            [-9999, 5, 5, 5, 5, 5, -9999],
            [-9999, 5, 5, 5, 5, 5, -9999],
            [-9999, 5, 5, 4, 5, 5, -9999],
            [-9999, 5, 5, 5, 5, 5, -9999],
            [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="expected_fdr")
def fixture_expected_fdr():
    """Create the expected_fdr for the test dem.

    Returns:
        np.ndarray: A dem of size 5x5.
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_EAST,
            ],
        ],
        dtype=np.uint8,
    )


def test_flow_direction_from_file(raster_file_path, expected_fdr):
    """Test the flow direction function from a raster file."""
    results_path = "/vsimem/test_flow_direction_results.tif"
    _flow_direction(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    fdr = band.ReadAsArray()
    assert np.array_equal(fdr, expected_fdr)


def test_flow_direction_from_dem(dem, expected_fdr):
    """Test the flow direction function from a raster file."""
    fdr = flow_direction_for_tile(dem, -9999)
    # remove buffer
    fdr = fdr[1:-1, 1:-1]
    assert np.array_equal(fdr, expected_fdr)


def test_flow_direction_cli(raster_file_path, expected_fdr):
    """Test the CLI."""
    output_path = "/vsimem/test_flow_direction_cli.tif"
    runner = click.testing.CliRunner()
    result = runner.invoke(
        flow_direction_cli,
        [
            "--input_file",
            raster_file_path,
            "--output_file",
            output_path,
            "--chunk_size",
            "5",
        ],
    )
    assert result.exit_code == 0
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    fdr = band.ReadAsArray()
    assert np.array_equal(fdr, expected_fdr)
    dataset = None
    gdal.Unlink(output_path)
