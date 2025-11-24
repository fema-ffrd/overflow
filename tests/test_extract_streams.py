import numpy as np
import pytest
from osgeo import gdal, ogr, osr

from overflow._extract_streams.core import _extract_streams_core
from overflow._extract_streams.tiled import _extract_streams_tiled
from overflow._util.constants import FLOW_ACCUMULATION_NODATA, FLOW_DIRECTION_NODATA


@pytest.fixture(name="test_rasters")
def fixture_test_rasters():
    """Fixture providing a single FDR-FAC pair for testing."""
    fdr = np.array(
        [
            [4, 6, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [6, 5, 4, 4, 4, 5],
            [5, 6, 5, 4, 4, 4],
            [6, 5, 6, 5, 4, 6],
            [5, 6, 5, 6, 5, 4],
        ],
        dtype=np.ubyte,
    )

    fac = np.array(
        [
            [1, 3, 2, 1, 1, 1],
            [8, 7, 3, 2, 2, 1],
            [1, 7, 6, 5, 2, 1],
            [9, 1, 5, 4, 3, 1],
            [1, 7, 1, 2, 1, 1],
            [9, 1, 4, 1, 3, 2],
        ],
        dtype=np.int64,
    )

    return {"fdr": fdr, "fac": fac}


@pytest.fixture(name="raster_file_paths")
def fixture_raster_file_paths(test_rasters):
    """Fixture creating temporary file paths for FDR and FAC arrays."""
    fdr_path = "/vsimem/test_fdr.tif"
    fac_path = "/vsimem/test_fac.tif"

    # Create FDR raster
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        fdr_path,
        test_rasters["fdr"].shape[1],
        test_rasters["fdr"].shape[0],
        1,
        gdal.GDT_Byte,
    )
    band = dataset.GetRasterBand(1)
    band.WriteArray(test_rasters["fdr"])
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, 1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.FlushCache()
    dataset = None

    # Create FAC raster
    dataset = driver.Create(
        fac_path,
        test_rasters["fac"].shape[1],
        test_rasters["fac"].shape[0],
        1,
        gdal.GDT_Int64,
    )
    band = dataset.GetRasterBand(1)
    band.WriteArray(test_rasters["fac"])
    band.SetNoDataValue(FLOW_ACCUMULATION_NODATA)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, 1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.FlushCache()
    dataset = None

    yield {"fdr": fdr_path, "fac": fac_path}

    gdal.Unlink(fdr_path)
    gdal.Unlink(fac_path)


def test_extract_streams(raster_file_paths, test_rasters):
    """Test the core extract_streams function."""
    output_dir = "/vsimem/output_streams_core"
    fdr_path = raster_file_paths["fdr"]
    fac_path = raster_file_paths["fac"]
    threshold = 5

    _extract_streams_core(fac_path, fdr_path, output_dir, threshold)

    streams_raster = f"{output_dir}/streams.tif"
    streams_vector = f"{output_dir}/streams.gpkg"

    ds = gdal.Open(streams_raster)
    assert ds is not None
    array = ds.ReadAsArray()
    assert array is not None
    assert array.shape == test_rasters["fac"].shape
    assert np.sum(array) == np.sum(test_rasters["fac"] >= threshold)

    ds = ogr.Open(streams_vector)
    assert ds is not None
    streams_layer = ds.GetLayerByName("streams")
    assert streams_layer is not None
    assert streams_layer.GetFeatureCount() > 0
    junctions_layer = ds.GetLayerByName("junctions")
    assert junctions_layer is not None
    assert junctions_layer.GetFeatureCount() > 0

    # Clean up
    gdal.Unlink(streams_raster)
    gdal.Unlink(streams_vector)


def test_extract_streams_tiled(raster_file_paths, test_rasters):
    """Test the tiled extract_streams_tiled function."""
    output_dir = "/vsimem/output_streams_tiled"
    fdr_path = raster_file_paths["fdr"]
    fac_path = raster_file_paths["fac"]
    threshold = 5
    chunk_size = 3

    _extract_streams_tiled(fac_path, fdr_path, output_dir, threshold, chunk_size)

    # Check if output files are created
    streams_raster = f"{output_dir}/streams.tif"
    streams_vector = f"{output_dir}/streams.gpkg"

    # Basic validation of output
    ds = gdal.Open(streams_raster)
    assert ds is not None
    array = ds.ReadAsArray()
    assert array is not None
    assert array.shape == test_rasters["fac"].shape
    assert np.sum(array) == np.sum(test_rasters["fac"] >= threshold)

    ds = ogr.Open(streams_vector)
    assert ds is not None
    streams_layer = ds.GetLayerByName("streams")
    assert streams_layer is not None
    assert streams_layer.GetFeatureCount() > 0
    junctions_layer = ds.GetLayerByName("junctions")
    assert junctions_layer is not None
    assert junctions_layer.GetFeatureCount() > 0

    # Clean up
    gdal.Unlink(streams_raster)
    gdal.Unlink(streams_vector)


def test_compare_core_and_tiled(raster_file_paths):
    """Compare results of core and tiled algorithms."""
    output_dir = "/vsimem/output_streams_compare"
    fdr_path = raster_file_paths["fdr"]
    fac_path = raster_file_paths["fac"]
    threshold = 5
    chunk_size = 3

    # Run core algorithm
    _extract_streams_core(fac_path, fdr_path, f"{output_dir}/core", threshold)

    # Run tiled algorithm
    _extract_streams_tiled(
        fac_path, fdr_path, f"{output_dir}/tiled", threshold, chunk_size
    )

    # Compare raster
    core_raster = f"{output_dir}/core/streams.tif"
    tiled_raster = f"{output_dir}/tiled/streams.tif"
    core_ds = gdal.Open(core_raster)
    tiled_ds = gdal.Open(tiled_raster)
    assert core_ds is not None
    assert tiled_ds is not None
    core_array = core_ds.ReadAsArray()
    tiled_array = tiled_ds.ReadAsArray()
    assert core_array is not None
    assert tiled_array is not None
    assert core_array.shape == tiled_array.shape
    assert np.array_equal(core_array, tiled_array)

    # Compare vector
    core_vector = f"{output_dir}/core/streams.gpkg"
    tiled_vector = f"{output_dir}/tiled/streams.gpkg"
    core_ds = ogr.Open(core_vector)
    tiled_ds = ogr.Open(tiled_vector)
    assert core_ds is not None
    assert tiled_ds is not None
    core_streams_layer = core_ds.GetLayerByName("streams")
    tiled_streams_layer = tiled_ds.GetLayerByName("streams")
    assert core_streams_layer is not None
    assert tiled_streams_layer is not None
    assert core_streams_layer.GetFeatureCount() == tiled_streams_layer.GetFeatureCount()
    core_junctions_layer = core_ds.GetLayerByName("junctions")
    tiled_junctions_layer = tiled_ds.GetLayerByName("junctions")
    assert core_junctions_layer is not None
    assert tiled_junctions_layer is not None
    assert (
        core_junctions_layer.GetFeatureCount()
        == tiled_junctions_layer.GetFeatureCount()
    )

    # Clean up
    gdal.Unlink(core_vector)
    gdal.Unlink(tiled_vector)
