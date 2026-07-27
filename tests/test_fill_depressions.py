import numpy as np
import pytest
from numba import typed, types
from osgeo import gdal

from overflow._fill_depressions.core import make_sides, priority_flood_tile
from overflow._fill_depressions.tiled import _fill_depressions_tiled
from overflow._fill_depressions.tiled.global_state import handle_corner, handle_edge


@pytest.fixture(name="dem_values")
def fixture_dem_values():
    """Create a small 5x5 raster for testing"""
    np.random.seed(32)
    return np.random.randint(10, 50, size=(5, 5)).astype(np.float32)


@pytest.fixture(name="expected_filled_dem_values")
def fixture_expected_filled_dem_values(dem_values):
    """Create a small 5x5 raster for testing"""
    filled_dem = dem_values.copy()
    filled_dem[1, 1] = 14
    filled_dem[2, 2] = 14
    filled_dem[2, 3] = 14
    return filled_dem


@pytest.fixture(name="expected_filled_dem_labels")
def fixture_expected_filled_dem_labels():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 3, 3, 3, 2],
        ]
    ).astype(int)


@pytest.fixture(name="dem_filepath")
def fixture_dem_filepath(dem_values):
    """Create a small 5x5 raster for testing"""
    filepath = "/vsimem/dem.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filepath, 5, 5, 1, gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(dem_values)
    band = None
    ds = None
    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="dem_values_with_nodata")
def fixture_dem_values_with_nodata():
    """Create a small 5x5 raster for testing"""
    dem = np.array(
        [
            [-9999, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, -9999, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
        ]
    ).astype(np.float32)
    return dem


@pytest.fixture(name="expected_filled_dem_values_with_nodata")
def fixture_expected_filled_dem_values_with_nodata():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [-9999, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 20, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
        ]
    ).astype(np.float32)


@pytest.fixture(name="expected_filled_dem_labels_with_nodata")
def fixture_expected_filled_dem_labels_with_nodata():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ]
    ).astype(int)


def test_fill_tile_top_left(
    dem_values: np.ndarray,
    expected_filled_dem_values: np.ndarray,
    expected_filled_dem_labels: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(top=True, left=True)
    labels, graph, max_label = priority_flood_tile(dem_values, sides)
    assert np.allclose(dem_values, expected_filled_dem_values)
    assert np.array_equal(labels, expected_filled_dem_labels)
    assert graph[(2, 3)] == 20.0
    assert graph[(1, 2)] == 15.0
    assert max_label == 4


def test_fill_tile_bottom_left(
    dem_values: np.ndarray,
    expected_filled_dem_values: np.ndarray,
    expected_filled_dem_labels: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(bottom=True, left=True)
    labels, graph, max_label = priority_flood_tile(dem_values, sides)
    assert np.allclose(dem_values, expected_filled_dem_values)
    assert np.array_equal(labels, expected_filled_dem_labels)
    assert graph[(2, 3)] == 20.0
    assert graph[(1, 2)] == 21.0
    assert graph[(1, 3)] == 15.0
    assert max_label == 4


def test_fill_tile_with_nodata_top(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(top=True)
    labels, graph, max_label = priority_flood_tile(
        dem_values_with_nodata, sides, fill_holes=True
    )
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert graph[(1, 2)] == -np.inf
    assert max_label == 3


def test_fill_tile_with_nodata_right(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(right=True)
    labels, graph, max_label = priority_flood_tile(
        dem_values_with_nodata, sides, fill_holes=True
    )
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert graph[(1, 2)] == 50
    assert max_label == 3


def test_fill_tile_with_nodata_no_edge(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides()
    labels, _, max_label = priority_flood_tile(
        dem_values_with_nodata, sides, fill_holes=True
    )
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert max_label == 3


def test_handle_edge():
    """Test handle_edge function produces expected graph"""
    dem_a = np.array([1, 2, 3, 4, 5])
    labels_a = np.array([2, 2, 3, 3, 2])
    dem_b = np.array([5, 4, 3, 2, 1])
    labels_b = np.array([5, 5, 6, 6, 5])
    graph = typed.Dict.empty(
        key_type=types.Tuple([types.int64, types.int64]),
        value_type=types.float32,
    )
    graph[(2, 5)] = 5
    no_data = -9999

    handle_edge(dem_a, labels_a, dem_b, labels_b, graph, no_data)

    expected_graph = {(2, 5): 4, (2, 6): 3, (3, 5): 4, (3, 6): 3}
    assert len(graph) == 4
    for key, value in expected_graph.items():
        assert graph[key] == value


def test_handle_corner():
    """Test handle_corner function produces expected graph"""
    elev_a = 5
    label_a = 2
    elev_b = 1
    label_b = 5
    graph = typed.Dict.empty(
        key_type=types.Tuple([types.int64, types.int64]),
        value_type=types.float32,
    )
    graph[2, 5] = 6
    expected_graph = {(2, 5): 5}
    handle_corner(elev_a, label_a, elev_b, label_b, graph, -9999)
    assert len(graph) == 1
    for key, value in expected_graph.items():
        assert graph[key] == value


def test_fill_depressions_tiled(dem_filepath, expected_filled_dem_values):
    """Test filling single tile"""
    working_dir = "/vsimem"
    chunk_size = 2
    output_filepath = "/vsimem/filled_dem.tif"
    _fill_depressions_tiled(dem_filepath, output_filepath, chunk_size, working_dir)
    ds = gdal.Open(output_filepath)
    band = ds.GetRasterBand(1)
    filled_dem = band.ReadAsArray()
    assert np.allclose(filled_dem, expected_filled_dem_values)
    band = None
    ds = None
    gdal.Unlink(output_filepath)


@pytest.mark.parametrize("transpose", [False, True], ids=["1xN grid", "Nx1 grid"])
def test_fill_depressions_tiled_degenerate_tile_grid(transpose):
    """Seams must be joined when the tile grid is a single tile row or column.

    Regression test. Joining seams by walking a 2x2 tile block over
    range(rows - 1) x range(cols - 1) iterates zero times on a 1xN or Nx1 tile
    grid, which leaves every seam unjoined. Nothing raises; the fill is simply
    wrong, and only for rasters that are smaller than chunk_size in one dimension
    and larger in the other.

    The DEM is a plateau at 10 holding a depression at elevation 1 that straddles
    the tile seam. Its only outlet is a pass at elevation 5 lying wholly inside the
    second tile, so filling the first tile's half correctly is impossible without
    the seam graph. Correct output raises the whole depression to 5; an unjoined
    seam raises the first tile's half to 10 instead.
    """
    chunk_size = 8
    dem = np.full((8, 16), 10.0, dtype=np.float32)
    dem[2:6, 5:11] = 1.0  # depression straddling the seam at column 8
    dem[3, 11:16] = 5.0  # spill channel reaching the right edge, inside tile 2
    if transpose:
        dem = np.ascontiguousarray(dem.T)

    dem_path = "/vsimem/degenerate_grid_dem.tif"
    output_path = "/vsimem/degenerate_grid_filled.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(dem_path, dem.shape[1], dem.shape[0], 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    band.WriteArray(dem)
    band.SetNoDataValue(-9999.0)
    dataset.FlushCache()
    band = None
    dataset = None

    try:
        _fill_depressions_tiled(dem_path, output_path, chunk_size, "/vsimem")
        ds = gdal.Open(output_path)
        filled = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        expected = dem.copy()
        if transpose:
            expected[5:11, 2:6] = 5.0
        else:
            expected[2:6, 5:11] = 5.0
        assert np.allclose(filled, expected)
    finally:
        for path in (dem_path, output_path):
            if gdal.VSIStatL(path) is not None:
                gdal.Unlink(path)
