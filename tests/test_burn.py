import click.testing
import numpy as np
import pytest
from osgeo import gdal

from overflow import burn
from overflow._util.union_find import UnionFind
from overflow.cli import burn_cli

NODATA = -9999.0
# chunk sizes chosen so the same region is split differently by each one, and so
# that some of them leave a partial tile hanging off the raster edge
TILED_CHUNK_SIZES = [3, 4, 5, 7, 8, 13, 64]


def write_raster(path, array, nodata, data_type):
    """Write a numpy array to an in-memory GeoTiff for testing."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, array.shape[1], array.shape[0], 1, data_type)
    dataset.SetGeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    dataset.FlushCache()
    band = None
    dataset = None
    return path


def unlink_if_exists(path):
    """Remove an in-memory raster, tolerating one that was never created."""
    if gdal.VSIStatL(path) is not None:
        gdal.Unlink(path)


def read_raster(path):
    """Read the first band of a raster into a numpy array."""
    dataset = gdal.Open(path)
    array = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    return array


@pytest.fixture(name="dem_path")
def fixture_dem_path():
    """Create a 20x20 DEM whose every cell holds a distinct elevation.

    Distinct values make each region's min, max and mean unambiguous. One cell
    inside a region is nodata so it can be checked for exclusion.

    Yields:
        str: Path to the DEM raster.
    """
    dem = np.arange(400, dtype=np.float32).reshape(20, 20)
    dem[2, 2] = NODATA
    path = "/vsimem/test_burn_dem.tif"
    write_raster(path, dem, NODATA, gdal.GDT_Float32)
    yield path
    gdal.Unlink(path)


@pytest.fixture(name="mask_path")
def fixture_mask_path():
    """Create a 20x20 classified mask with regions that straddle tile seams.

    The regions are, by class:
      1: a wide band across rows 1-3, split by every vertical seam under test
      1: a blob at rows 9-12, split by horizontal seams and disconnected from the band
      1: two diagonally touching cells, one region under 8-connectivity and two
         under 4-connectivity
      2: an L whose two arms only meet by passing through other tiles

    Yields:
        str: Path to the mask raster.
    """
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:4, 1:12] = 1
    mask[6:8, 2:16] = 2
    mask[6:16, 14:16] = 2
    mask[9:13, 16:18] = 1
    mask[7, 7] = 1
    mask[8, 8] = 1
    path = "/vsimem/test_burn_mask.tif"
    write_raster(path, mask, 0.0, gdal.GDT_Byte)
    yield path
    gdal.Unlink(path)


@pytest.fixture(name="output_path")
def fixture_output_path():
    """Provide a scratch output path that is cleaned up after the test.

    Yields:
        str: Path for the burned DEM.
    """
    path = "/vsimem/test_burn_output.tif"
    yield path
    unlink_if_exists(path)


@pytest.fixture(name="small_dem_path")
def fixture_small_dem_path():
    """Create a 5x5 DEM with a simple gradient.

    Yields:
        str: Path to the DEM raster.
    """
    dem = np.array(
        [
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
        ],
        dtype=np.float32,
    )
    path = "/vsimem/test_burn_small_dem.tif"
    write_raster(path, dem, NODATA, gdal.GDT_Float32)
    yield path
    gdal.Unlink(path)


@pytest.fixture(name="small_mask_path")
def fixture_small_mask_path():
    """Create a 5x5 mask holding one class 1 region and one class 3 region.

    Yields:
        str: Path to the mask raster.
    """
    mask = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 3],
            [0, 0, 0, 3, 3],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    path = "/vsimem/test_burn_small_mask.tif"
    write_raster(path, mask, 0.0, gdal.GDT_Byte)
    yield path
    gdal.Unlink(path)


def test_burn_constant(small_dem_path, small_mask_path, output_path):
    """Each mask class is set to its own constant elevation."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="constant",
        burn_values="1:100,3:200",
        chunk_size=1,
    )
    expected = np.array(
        [
            [10, 100, 100, 40, 50],
            [11, 100, 31, 41, 200],
            [12, 22, 32, 200, 200],
            [100, 23, 33, 43, 53],
            [100, 100, 34, 44, 54],
        ],
        dtype=np.float32,
    )
    assert np.allclose(read_raster(output_path), expected)


def test_burn_relative(small_dem_path, small_mask_path, output_path):
    """Relative burning subtracts per cell, preserving relief inside a region."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="relative",
        burn_values="1:10,3:20",
        chunk_size=1,
    )
    expected = np.array(
        [
            [10, 10, 20, 40, 50],
            [11, 11, 31, 41, 31],
            [12, 22, 32, 22, 32],
            [3, 23, 33, 43, 53],
            [4, 14, 34, 44, 54],
        ],
        dtype=np.float32,
    )
    assert np.allclose(read_raster(output_path), expected)


def test_burn_relative_broadcast(small_dem_path, small_mask_path, output_path):
    """A bare burn value applies to every selected mask value."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="relative",
        burn_values="2.5",
        chunk_size=1,
    )
    result = read_raster(output_path)
    mask = read_raster(small_mask_path)
    dem = read_raster(small_dem_path)
    assert np.allclose(result[mask != 0], dem[mask != 0] - 2.5)
    assert np.allclose(result[mask == 0], dem[mask == 0])


@pytest.mark.parametrize(
    "statistic,expected_region_1,expected_region_3",
    [("min", 20.0, 42.0), ("max", 30.0, 52.0), ("mean", 71.0 / 3, 145.0 / 3)],
)
def test_burn_statistic(
    small_dem_path,
    small_mask_path,
    output_path,
    statistic,
    expected_region_1,
    expected_region_3,
):
    """Each contiguous region is flattened to its own statistic."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="statistic",
        mask_values="1,3",
        statistic=statistic,
        chunk_size=1,
    )
    result = read_raster(output_path)
    # the upper class 1 region: DEM values 20, 30, 21
    assert np.allclose(result[0, 1], expected_region_1)
    assert np.allclose(result[0, 2], expected_region_1)
    assert np.allclose(result[1, 1], expected_region_1)
    # the class 3 region: DEM values 51, 42, 52
    assert np.allclose(result[1, 4], expected_region_3)
    assert np.allclose(result[2, 3], expected_region_3)
    assert np.allclose(result[2, 4], expected_region_3)
    # the lower class 1 region is a separate region and keeps its own statistic
    assert not np.allclose(result[3, 0], expected_region_1)


def test_burn_statistic_offset(small_dem_path, small_mask_path, output_path):
    """burn_offset is subtracted from each region's statistic."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="statistic",
        mask_values="1,3",
        statistic="min",
        burn_offset=1.5,
        chunk_size=1,
    )
    result = read_raster(output_path)
    assert np.allclose(result[0, 1], 20.0 - 1.5)
    assert np.allclose(result[2, 3], 42.0 - 1.5)


def test_burn_statistic_leaves_unselected_cells_alone(
    small_dem_path, small_mask_path, output_path
):
    """Cells outside the selected mask values are copied through unchanged."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="statistic",
        mask_values="3",
        statistic="min",
        chunk_size=1,
    )
    result = read_raster(output_path)
    dem = read_raster(small_dem_path)
    mask = read_raster(small_mask_path)
    assert np.allclose(result[mask != 3], dem[mask != 3])
    assert np.allclose(result[mask == 3], 42.0)


def test_burn_default_mask_values_keeps_classes_separate(
    small_dem_path, small_mask_path, output_path
):
    """Omitting mask values selects every non zero class, each still its own region."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="statistic",
        statistic="min",
        chunk_size=1,
    )
    result = read_raster(output_path)
    assert np.allclose(result[0, 1], 20.0)
    assert np.allclose(result[2, 3], 42.0)
    assert np.allclose(result[4, 0], 13.0)


@pytest.mark.parametrize("connectivity,expected", [(8, 1), (4, 2)])
def test_burn_connectivity(dem_path, output_path, connectivity, expected):
    """Diagonally touching cells are one region under 8-connectivity, two under 4."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    mask[6, 6] = 1
    mask_path = "/vsimem/test_burn_diagonal_mask.tif"
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="min",
            connectivity=connectivity,
            chunk_size=1,
        )
        result = read_raster(output_path)
        distinct = {result[5, 5], result[6, 6]}
        assert len(distinct) == expected
        if expected == 1:
            # joined: both take the minimum of the pair
            assert np.allclose(result[5, 5], 105.0)
            assert np.allclose(result[6, 6], 105.0)
        else:
            # separate: each keeps its own value
            assert np.allclose(result[5, 5], 105.0)
            assert np.allclose(result[6, 6], 126.0)
    finally:
        gdal.Unlink(mask_path)


@pytest.mark.parametrize("chunk_size", TILED_CHUNK_SIZES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "statistic", "mask_values": "1,2", "statistic": "min"},
        {"method": "statistic", "mask_values": "1,2", "statistic": "max"},
        {"method": "statistic", "mask_values": "1,2", "statistic": "mean"},
        {
            "method": "statistic",
            "mask_values": "1,2",
            "statistic": "min",
            "burn_offset": 5.0,
        },
        {
            "method": "statistic",
            "mask_values": "1,2",
            "statistic": "min",
            "connectivity": 4,
        },
        {"method": "constant", "burn_values": "1:100,2:200"},
        {"method": "relative", "burn_values": "1:10,2:20"},
    ],
    ids=["min", "max", "mean", "offset", "connectivity4", "constant", "relative"],
)
def test_compare_core_and_tiled(dem_path, mask_path, chunk_size, kwargs):
    """Tiled processing matches in-memory processing at every tile size.

    The mask fixture holds regions that straddle vertical seams, horizontal seams
    and the point where four tiles meet, so this is the check that the cross tile
    region solve actually works.
    """
    core_path = "/vsimem/test_burn_core.tif"
    tiled_path = "/vsimem/test_burn_tiled.tif"
    try:
        burn(dem_path, mask_path, core_path, chunk_size=1, **kwargs)
        burn(dem_path, mask_path, tiled_path, chunk_size=chunk_size, **kwargs)
        assert np.allclose(read_raster(core_path), read_raster(tiled_path))
    finally:
        gdal.Unlink(core_path)
        gdal.Unlink(tiled_path)


@pytest.mark.parametrize("chunk_size", [4, 5, 8])
def test_burn_tiled_region_spanning_seam(chunk_size, output_path):
    """A region split across tiles gets one statistic, not one per fragment."""
    # a single 16 cell horizontal bar that every tested chunk size cuts in two
    dem = np.arange(64, dtype=np.float32).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3, :] = 1
    dem_path = "/vsimem/test_burn_seam_dem.tif"
    mask_path = "/vsimem/test_burn_seam_mask.tif"
    write_raster(dem_path, dem, NODATA, gdal.GDT_Float32)
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="min",
            chunk_size=chunk_size,
        )
        result = read_raster(output_path)
        # the whole bar takes the minimum of the whole bar, dem[3, 0] == 24
        assert np.allclose(result[3, :], 24.0)
    finally:
        gdal.Unlink(dem_path)
        gdal.Unlink(mask_path)


@pytest.mark.parametrize("shape", [(4, 32), (32, 4)])
def test_burn_tiled_single_tile_row_or_column(shape, output_path):
    """Rasters whose tile grid is a single row or column still join their seams.

    A tile grid of 1xN or Nx1 is the case a 2x2 tile stencil would silently skip.
    """
    rows, cols = shape
    dem = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    mask = np.ones((rows, cols), dtype=np.uint8)
    dem_path = "/vsimem/test_burn_strip_dem.tif"
    mask_path = "/vsimem/test_burn_strip_mask.tif"
    write_raster(dem_path, dem, NODATA, gdal.GDT_Float32)
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="max",
            chunk_size=8,
        )
        # the mask covers the whole raster, so it is one region with one maximum
        assert np.allclose(read_raster(output_path), float(rows * cols - 1))
    finally:
        gdal.Unlink(dem_path)
        gdal.Unlink(mask_path)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_burn_dem_nodata_excluded_and_preserved(chunk_size, output_path):
    """DEM nodata cells are excluded from statistics and never written."""
    dem = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, NODATA, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    dem_path = "/vsimem/test_burn_nodata_dem.tif"
    mask_path = "/vsimem/test_burn_nodata_mask.tif"
    write_raster(dem_path, dem, NODATA, gdal.GDT_Float32)
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="mean",
            chunk_size=chunk_size,
        )
        result = read_raster(output_path)
        # mean over 1, 2, 5 only; the nodata cell contributes nothing
        assert np.allclose(result[0, 0], 8.0 / 3)
        assert np.allclose(result[0, 1], 8.0 / 3)
        assert np.allclose(result[1, 0], 8.0 / 3)
        assert result[1, 1] == NODATA
    finally:
        gdal.Unlink(dem_path)
        gdal.Unlink(mask_path)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_burn_region_entirely_nodata_is_unchanged(chunk_size, output_path):
    """A region with no valid DEM cells is left alone rather than filled with junk."""
    dem = np.arange(16, dtype=np.float32).reshape(4, 4)
    dem[3, 3] = NODATA
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[3, 3] = 1
    mask[0, 0] = 1
    dem_path = "/vsimem/test_burn_empty_dem.tif"
    mask_path = "/vsimem/test_burn_empty_mask.tif"
    write_raster(dem_path, dem, NODATA, gdal.GDT_Float32)
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="min",
            chunk_size=chunk_size,
        )
        assert np.allclose(read_raster(output_path), dem)
    finally:
        gdal.Unlink(dem_path)
        gdal.Unlink(mask_path)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_burn_mask_without_nodata_value(chunk_size, output_path):
    """A mask raster that declares no nodata value still works."""
    dem = np.arange(36, dtype=np.float32).reshape(6, 6)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    dem_path = "/vsimem/test_burn_nonodata_dem.tif"
    mask_path = "/vsimem/test_burn_nonodata_mask.tif"
    write_raster(dem_path, dem, NODATA, gdal.GDT_Float32)
    write_raster(mask_path, mask, None, gdal.GDT_Byte)
    try:
        burn(
            dem_path,
            mask_path,
            output_path,
            method="statistic",
            mask_values="1",
            statistic="min",
            chunk_size=chunk_size,
        )
        result = read_raster(output_path)
        assert np.allclose(result[2:5, 2:5], 14.0)
        assert np.allclose(result[0, :], dem[0, :])
    finally:
        gdal.Unlink(dem_path)
        gdal.Unlink(mask_path)


def test_burn_output_preserves_geotransform_and_nodata(
    small_dem_path, small_mask_path, output_path
):
    """The output carries the input DEM's georeferencing, data type and nodata."""
    burn(
        small_dem_path,
        small_mask_path,
        output_path,
        method="constant",
        burn_values="1:1,3:1",
        chunk_size=1,
    )
    source = gdal.Open(small_dem_path)
    result = gdal.Open(output_path)
    assert result.GetGeoTransform() == source.GetGeoTransform()
    assert result.GetRasterBand(1).DataType == source.GetRasterBand(1).DataType
    assert result.GetRasterBand(1).GetNoDataValue() == NODATA
    source = None
    result = None


def test_burn_rejects_incompatible_rasters(small_dem_path, output_path):
    """A mask that is not co-registered with the DEM is rejected."""
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask_path = "/vsimem/test_burn_bad_shape_mask.tif"
    write_raster(mask_path, mask, 0.0, gdal.GDT_Byte)
    try:
        with pytest.raises(ValueError):
            burn(
                small_dem_path,
                mask_path,
                output_path,
                method="statistic",
                mask_values="1",
                chunk_size=1,
            )
    finally:
        gdal.Unlink(mask_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "nonsense", "burn_values": "1:1"},
        {"method": "statistic", "statistic": "median"},
        {"method": "statistic", "connectivity": 6},
        {"method": "constant"},
        {"method": "relative"},
        {"method": "constant", "burn_values": "1:oops"},
        {"method": "constant", "burn_values": "1:2:3"},
        {"method": "constant", "burn_values": "1:2", "mask_values": "1,3"},
        {"method": "statistic", "mask_values": "1,oops"},
        {"method": "statistic", "mask_values": "0"},
    ],
    ids=[
        "bad_method",
        "bad_statistic",
        "bad_connectivity",
        "constant_without_burn_values",
        "relative_without_burn_values",
        "non_numeric_burn_value",
        "malformed_burn_value",
        "missing_burn_value_for_class",
        "non_numeric_mask_value",
        "mask_value_is_nodata",
    ],
)
def test_burn_rejects_bad_arguments(
    small_dem_path, small_mask_path, output_path, kwargs
):
    """Malformed or inconsistent arguments raise before any raster work starts."""
    with pytest.raises(ValueError):
        burn(small_dem_path, small_mask_path, output_path, chunk_size=1, **kwargs)


def test_burn_cli(small_dem_path, small_mask_path, output_path):
    """The CLI runs the statistic method end to end."""
    runner = click.testing.CliRunner()
    result = runner.invoke(
        burn_cli,
        [
            "--dem_file",
            small_dem_path,
            "--mask_file",
            small_mask_path,
            "--output_file",
            output_path,
            "--method",
            "statistic",
            "--mask_values",
            "1,3",
            "--statistic",
            "min",
            "--burn_offset",
            "1.0",
            "--chunk_size",
            "1",
        ],
    )
    assert result.exit_code == 0
    burned = read_raster(output_path)
    assert np.allclose(burned[2, 3], 41.0)


def test_burn_cli_constant(small_dem_path, small_mask_path, output_path):
    """The CLI accepts the paired burn value mapping for the constant method."""
    runner = click.testing.CliRunner()
    result = runner.invoke(
        burn_cli,
        [
            "--dem_file",
            small_dem_path,
            "--mask_file",
            small_mask_path,
            "--output_file",
            output_path,
            "--method",
            "constant",
            "--burn_values",
            "1:100,3:200",
            "--chunk_size",
            "2",
        ],
    )
    assert result.exit_code == 0
    burned = read_raster(output_path)
    assert np.allclose(burned[0, 1], 100.0)
    assert np.allclose(burned[2, 3], 200.0)


def test_union_find_joins_and_separates():
    """UnionFind reports one representative per connected group."""
    union_find = UnionFind()
    union_find.union(1, 2)
    union_find.union(2, 3)
    union_find.union(10, 11)
    assert union_find.find(1) == union_find.find(3)
    assert union_find.find(10) == union_find.find(11)
    assert union_find.find(1) != union_find.find(10)
    # an unseen key is its own representative
    assert union_find.find(99) == 99


def test_union_find_merges_long_chains():
    """Chained unions collapse to a single representative."""
    union_find = UnionFind()
    for i in range(1, 200):
        union_find.union(i, i + 1)
    roots = {union_find.find(i) for i in range(1, 201)}
    assert len(roots) == 1
