import numpy as np
import pytest
from numba import int64  # type: ignore[attr-defined]
from numba.typed import Dict  # type: ignore[attr-defined]
from numba.types import UniTuple
from osgeo import gdal, ogr, osr

from overflow.basins.core import label_watersheds
from overflow.basins.tiled import label_watersheds_tiled
from overflow.util.constants import FLOW_DIRECTION_NODATA


@pytest.fixture(name="test_fdr")
def fixture_test_fdr():
    """Create a sample FDR for testing."""
    fdr = np.array(
        [
            [3, 4, 0, 1],
            [2, 3, 1, 2],
            [6, 5, 7, 6],
            [5, 4, 0, 7],
        ],
        dtype=np.ubyte,
    )
    return fdr


@pytest.fixture(name="expected_watersheds")
def fixture_expected_watersheds():
    """The expected watersheds from the test fdr."""
    watersheds = np.array(
        [
            [1, 1, 4, 4],
            [1, 1, 4, 4],
            [13, 13, 16, 16],
            [13, 13, 16, 16],
        ],
        dtype=np.int64,
    )
    return watersheds


@pytest.fixture(name="test_fdr_with_drainage_points")
def fixture_test_fdr_with_drainage_points():
    """Create a sample FDR for testing with drainage points."""
    fdr = np.array(
        [
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [0, 0, 6, 4, 4],
        ],
        dtype=np.ubyte,
    )
    return fdr


@pytest.fixture(name="test_fdr_with_drainage_points_filepath")
def fixture_test_fdr_with_drainage_points_filepath(test_fdr_with_drainage_points):
    """Create a filepath for test_fdr_with_drainage_points."""
    filepath = "/vsimem/test_fdr.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = test_fdr_with_drainage_points.shape
    dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, 1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    band.WriteArray(test_fdr_with_drainage_points)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="test_drainage_points")
def fixture_test_drainage_points():
    """Create a sample drainage points."""
    drainage_points = Dict.empty(UniTuple(int64, 2), int64)
    drainage_points[(2, 2)] = 0
    drainage_points[(3, 2)] = 0
    return drainage_points


@pytest.fixture(name="expected_watersheds_with_drainage_points")
def fixture_expected_watersheds_with_drainage_points():
    """The expected watersheds from the test_fdr_with_drainage_points."""
    watersheds = np.array(
        [
            [13, 13, 13, 13, 13],
            [18, 13, 13, 13, 18],
            [23, 18, 13, 18, 23],
            [23, 23, 18, 23, 23],
            [23, 23, 23, 23, 23],
        ],
        dtype=np.int64,
    )
    return watersheds


def test_label_watersheds(test_fdr, expected_watersheds):
    """Test the label_watersheds function."""
    drainage_points = Dict.empty(UniTuple(int64, 2), int64)
    watersheds, local_graph = label_watersheds(test_fdr, drainage_points)
    assert np.array_equal(watersheds, expected_watersheds)
    assert len(local_graph) == 0


def test_label_watersheds_with_drainage_points(
    test_fdr_with_drainage_points,
    test_drainage_points,
    expected_watersheds_with_drainage_points,
):
    """Test the label_watersheds function with drainage points."""
    watersheds, local_graph = label_watersheds(
        test_fdr_with_drainage_points, test_drainage_points
    )
    assert np.array_equal(watersheds, expected_watersheds_with_drainage_points)
    assert local_graph[18] == 23
    assert local_graph[13] == 18
    assert test_drainage_points[(2, 2)] == 13
    assert test_drainage_points[(3, 2)] == 18


def test_label_watersheds_tiled(
    test_fdr_with_drainage_points_filepath,
    test_drainage_points,
    expected_watersheds_with_drainage_points,
):
    """Test the label_watersheds_tiled function."""
    output_path = "/vsimem/watersheds.tif"
    graph = label_watersheds_tiled(
        test_fdr_with_drainage_points_filepath,
        test_drainage_points,
        output_path,
        chunk_size=3,  # chunk size that does not evenly divide the raster
    )
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    watersheds = band.ReadAsArray()
    # Remap the watershed ids to match the expected watersheds
    watershed_id_map = {
        9: 13,
        21: 18,
        24: 23,
    }
    for old_id, new_id in watershed_id_map.items():
        watersheds = np.where(watersheds == old_id, new_id, watersheds)
    assert np.array_equal(watersheds, expected_watersheds_with_drainage_points)
    assert graph[21] == 24
    assert graph[9] == 21
    assert test_drainage_points[(2, 2)] == 9
    assert test_drainage_points[(3, 2)] == 21

    # check for basins gpkg file
    gpkg_path = "/vsimem/watersheds.gpkg"
    ds = ogr.Open(gpkg_path)
    assert ds is not None
    layer = ds.GetLayer()
    assert layer.GetFeatureCount() == 3
    ds = None
    gdal.Unlink(output_path)
    gdal.Unlink(gpkg_path)
