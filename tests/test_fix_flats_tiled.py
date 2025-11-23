import numpy as np
import pytest
from osgeo import gdal

from overflow.resolve_flats.tiled.resolve_flats_tiled import _resolve_flats_tiled
from overflow.util.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_WEST,
)


@pytest.fixture(name="dem_zhou_2022")
def fixture_dem_zhou_2022():
    """DEM from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [5, 4, 4, 5, 9, 7, 2, 4],
            [9, 3, 3, 3, 3, 3, 3, 7],
            [7, 3, 3, 3, 3, 3, 3, 5],
            [8, 3, 3, 3, 3, 3, 3, 6],
            [9, 3, 3, 3, 3, 3, 3, 2],
            [6, 3, 3, 3, 3, 3, 3, 6],
            [5, 3, 3, 3, 3, 3, 3, 5],
            [1, 8, 9, 5, 6, 6, 7, 4],
        ],
        np.uint32,
    )


@pytest.fixture(name="dem_zhou_2022_filepath")
def fixture_dem_zhou_2022_filepath(dem_zhou_2022):
    """DEM from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a DEM file
    """
    dem_path = "/vsimem/dem_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = dem_zhou_2022.shape
    dataset = driver.Create(dem_path, cols, rows, 1, gdal.GDT_Float32)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset.GetRasterBand(1).WriteArray(dem_zhou_2022)
    dataset = None
    yield dem_path
    gdal.Unlink(dem_path)


@pytest.fixture(name="fdr_zhou_2022")
def fixture_fdr_zhou_2022():
    """FDR from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_EAST,
            ],
        ],
        np.uint32,
    )


@pytest.fixture(name="fdr_zhou_2022_filepath")
def fixture_fdr_zhou_2022_filepath(fdr_zhou_2022):
    """FDR from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a FDR file
    """
    fdr_path = "/vsimem/fdr_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = fdr_zhou_2022.shape
    dataset = driver.Create(fdr_path, cols, rows, 1, gdal.GDT_Byte)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(FLOW_DIRECTION_NODATA)
    dataset.GetRasterBand(1).WriteArray(fdr_zhou_2022)
    dataset = None
    yield fdr_path
    gdal.Unlink(fdr_path)


@pytest.fixture(name="expected_fixed_fdr_zhou_2022")
def fixture_expected_fixed_fdr_zhou_2022():
    """Create the expected fixed fdr for the test dem.

    Returns:
        np.ndarray: A 2D array containing the expected fixed fdr
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_EAST,
            ],
        ],
        np.uint32,
    )


@pytest.mark.parametrize("chunk_size", [2, 4, 8, 12])
def test__resolve_flats_tiled(
    dem_zhou_2022_filepath,
    fdr_zhou_2022_filepath,
    expected_fixed_fdr_zhou_2022,
    chunk_size,
):
    """
    Test the fix_flats function with different chunk sizes.
     - Chunk size 2 will contain tiles that contain only flats
     - Chunk size 4 will test the example in the paper
     - Chunk size 8 will test a single tile the size of the entire DEM
     - Chunk size 12 will test a chunk size larger than the DEM
    """
    output_filepath = f"/vsimem/fixed_fdr_{chunk_size}.tif"
    _resolve_flats_tiled(
        dem_zhou_2022_filepath,
        fdr_zhou_2022_filepath,
        output_filepath,
        chunk_size=chunk_size,
        working_dir=f"/vsimem/{chunk_size}/",
    )
    fixed_fdr_dataset = gdal.Open(output_filepath)
    fixed_fdr = fixed_fdr_dataset.GetRasterBand(1).ReadAsArray()
    flat_mask_dataset = gdal.Open(f"/vsimem/{chunk_size}/flat_mask.tif")
    flat_mask_dataset.GetRasterBand(1).ReadAsArray()
    assert np.array_equal(fixed_fdr, expected_fixed_fdr_zhou_2022)
