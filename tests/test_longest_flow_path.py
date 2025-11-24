import math

import numpy as np
import pytest
from numba.core import types
from numba.typed import Dict, List
from numba.types import int64
from osgeo import gdal, ogr, osr

from overflow._longest_flow_path import _flow_length_core
from overflow._longest_flow_path.core.longest_flow_path import (
    calculate_flow_distance_geographic,
    calculate_flow_distance_projected,
    calculate_path_distance_geographic,
    calculate_path_distance_projected,
    calculate_upstream_flow_length,
    cell_to_geographic_coords,
    create_longest_flow_path_vectors,
    find_all_upstream_basins,
    haversine_distance,
    is_geographic,
    projected_step_distance,
    trace_longest_flow_path,
    trace_path_from_cell,
)
from overflow._util.constants import FLOW_DIRECTION_NODATA

# =============================================================================
# FDR Fixtures
# =============================================================================


@pytest.fixture(name="single_outlet_fdr")
def fixture_single_outlet_fdr():
    """
    Flow direction raster with a single outlet at (4, 2).

    Flow directions (D8 encoding):
        0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE

    Grid (5x5) - all cells drain to top-left in a zig-zag pattern:
    """
    return np.array(
        [
            [4, 4, 4, 4, 4],
            [0, 0, 0, 0, 2],
            [2, 4, 4, 4, 4],
            [0, 0, 0, 0, 2],
            [2, 4, 4, 4, 4],
        ],
        dtype=np.ubyte,
    )


@pytest.fixture(name="two_outlet_fdr")
def fixture_two_outlet_fdr():
    """
    Flow direction raster with two separate outlets.

    Grid (5x6) - left side drains to (4,0), right side drains to (4,5):
        6  6  6 | 6  6  6
        6  6  6 | 6  6  6
        6  6  6 | 6  6  6
        6  5  6 | 6  7  6
        X  4  4 | 0  0  X   <- X = outlets at (4,0) and (4,5)

    Left basin: columns 0-2, outlet at (4,0)
    Right basin: columns 3-5, outlet at (4,5)
    """
    return np.array(
        [
            [6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6],
            [6, 5, 6, 6, 7, 6],
            [9, 4, 4, 0, 0, 9],
        ],
        dtype=np.ubyte,
    )


# =============================================================================
# File-based Fixtures
# =============================================================================


@pytest.fixture(name="single_outlet_fdr_file")
def fixture_single_outlet_fdr_file(single_outlet_fdr):
    """GeoTIFF file for single_outlet_fdr with 10m pixels."""
    filepath = "/vsimem/single_outlet_fdr.tif"
    rows, cols = single_outlet_fdr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 10, 0, 50, 0, -10])  # 10m pixels
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)  # UTM Zone 10N
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    band.WriteArray(single_outlet_fdr)
    ds.FlushCache()
    ds = None
    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="two_outlet_fdr_file")
def fixture_two_outlet_fdr_file(two_outlet_fdr):
    """GeoTIFF file for single_outlet_fdr with 10m pixels."""
    filepath = "/vsimem/two_outlet_fdr.tif"
    rows, cols = two_outlet_fdr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 10, 0, 50, 0, -10])  # 10m pixels
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)  # UTM Zone 10N
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    band.WriteArray(two_outlet_fdr)
    ds.FlushCache()
    ds = None
    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="single_drainage_point_file")
def fixture_single_drainage_point_file():
    """GeoPackage with single drainage point at (0,0). Map coords (5,45)."""
    filepath = "/vsimem/single_dp.gpkg"
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(filepath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    layer = ds.CreateLayer("points", srs, ogr.wkbPoint)

    # Cell (0, 0) with 10m pixels, origin at (0, 50): x=5, y=45
    feat = ogr.Feature(layer.GetLayerDefn())
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(5, 45)
    feat.SetGeometry(pt)
    feat.SetFID(1)
    layer.CreateFeature(feat)
    ds = None

    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="multi_drainage_point_file")
def fixture_multi_drainage_point_file():
    """
    GeoPackage with drainage points at
        - (4,0) Map coords (5,5).
        - (4,5) Map coords (55,5).
        - (3,2) Map coords (25,15).
    """
    filepath = "/vsimem/single_dp.gpkg"
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(filepath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    layer = ds.CreateLayer("points", srs, ogr.wkbPoint)

    # Cell (4, 0) with 10m pixels, origin at (0, 50): x=5, y=5
    feat = ogr.Feature(layer.GetLayerDefn())
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(5, 5)
    feat.SetGeometry(pt)
    feat.SetFID(1)
    layer.CreateFeature(feat)

    # Cell (4, 5) with 10m pixels, origin at (0, 50): x=55, y=5
    feat = ogr.Feature(layer.GetLayerDefn())
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(55, 5)
    feat.SetGeometry(pt)
    feat.SetFID(2)
    layer.CreateFeature(feat)

    # Cell (3, 2) with 10m pixels, origin at (0, 50): x=25, y=15
    feat = ogr.Feature(layer.GetLayerDefn())
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(25, 15)
    feat.SetGeometry(pt)
    feat.SetFID(3)
    layer.CreateFeature(feat)

    ds = None

    yield filepath
    gdal.Unlink(filepath)


# =============================================================================
# Tests
# =============================================================================


def lines_to_paths(lines_filepath: str) -> dict[int, list[tuple[int, int]]]:
    """Convert a linestring GeoPackage to a list of (row, col) tuples."""
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.Open(lines_filepath)
    layer = ds.GetLayer(0)
    paths = {}
    for feature in layer:
        geom = feature.GetGeometryRef()
        dp_id = feature.GetField("dp_id")
        path = []
        for i in range(geom.GetPointCount()):
            x, y, _ = geom.GetPoint(i)
            col = int(x // 10)
            row = int((50 - y) // 10)
            path.append((row, col))
        paths[dp_id] = path
    ds = None
    return paths


def read_flow_length(flow_length_filepath: str) -> np.ndarray:
    """Read flow length raster from GeoTIFF file."""
    ds = gdal.Open(flow_length_filepath)
    band = ds.GetRasterBand(1)
    flow_length = band.ReadAsArray()
    ds = None
    return flow_length  # type: ignore[no-any-return]


def test_single_outlet_longest_flow_path(
    single_outlet_fdr_file, single_drainage_point_file
):
    _flow_length_core(
        single_outlet_fdr_file,
        single_drainage_point_file,
        output_raster="/vsimem/output.tif",
        output_vector="/vsimem/output_paths.gpkg",
        layer_name="points",
    )
    paths = lines_to_paths("/vsimem/output_paths.gpkg")
    expected_path = [
        (4, 4),
        (4, 3),
        (4, 2),
        (4, 1),
        (4, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (2, 4),
        (2, 3),
        (2, 2),
        (2, 1),
        (2, 0),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
    ]
    assert len(paths) == 1
    assert 1 in paths
    assert paths[1] == expected_path
    expected_flow_length = np.array(
        [
            0,
            10,
            20,
            30,
            40,
            90,
            80,
            70,
            60,
            50,
            100,
            110,
            120,
            130,
            140,
            190,
            180,
            170,
            160,
            150,
            200,
            210,
            220,
            230,
            240,
        ]
    ).reshape((5, 5))
    flow_length = read_flow_length("/vsimem/output.tif")
    np.testing.assert_array_almost_equal(flow_length, expected_flow_length)


def test_two_outlet_longest_flow_path(two_outlet_fdr_file, multi_drainage_point_file):
    _flow_length_core(
        two_outlet_fdr_file,
        multi_drainage_point_file,
        output_raster="/vsimem/output.tif",
        output_vector="/vsimem/output_paths.gpkg",
        layer_name="points",
    )
    paths = lines_to_paths("/vsimem/output_paths.gpkg")

    expected_path_1 = [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (4, 1),
        (4, 0),
    ]
    expected_path_2 = [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (4, 4),
        (4, 5),
    ]
    expected_path_3 = [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
    ]

    assert len(paths) == 3
    assert 1 in paths
    assert 2 in paths
    assert 3 in paths
    assert paths[1] == expected_path_1
    assert paths[2] == expected_path_2
    assert paths[3] == expected_path_3

    expected_flow_length = np.array(
        [
            [40, 10 * math.sqrt(2) + 30, 30, 60, 10 * math.sqrt(2) + 30, 40],
            [30, 10 * math.sqrt(2) + 20, 20, 50, 10 * math.sqrt(2) + 20, 30],
            [20, 10 * math.sqrt(2) + 10, 10, 40, 10 * math.sqrt(2) + 10, 20],
            [10, 10 * math.sqrt(2), 0, 30, 10 * math.sqrt(2), 10],
            [0, 10, 20, 20, 10, 0],
        ]
    )
    flow_length = read_flow_length("/vsimem/output.tif")
    np.testing.assert_array_almost_equal(flow_length, expected_flow_length)


# =============================================================================
# Unit Tests for Distance Calculation Primitives
# =============================================================================


class TestCellToGeographicCoords:
    """Tests for cell_to_geographic_coords function."""

    def test_origin_cell(self):
        """Cell (0,0) should map to top-left corner + 0.5 pixel offset."""
        # geotransform: (origin_x, pixel_width, rotation, origin_y, rotation, -pixel_height)
        gt = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)
        lon, lat = cell_to_geographic_coords(0, 0, gt)
        assert lon == pytest.approx(5.0)  # 0 + 0.5 * 10
        assert lat == pytest.approx(95.0)  # 100 + 0.5 * -10

    def test_offset_cell(self):
        """Cell (2, 3) should be offset correctly from origin."""
        gt = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)
        lon, lat = cell_to_geographic_coords(2, 3, gt)
        assert lon == pytest.approx(35.0)  # 0 + (3 + 0.5) * 10
        assert lat == pytest.approx(75.0)  # 100 + (2 + 0.5) * -10

    def test_with_non_zero_origin(self):
        """Test with non-zero origin coordinates."""
        gt = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
        lon, lat = cell_to_geographic_coords(1, 1, gt)
        assert lon == pytest.approx(500045.0)  # 500000 + 1.5 * 30
        assert lat == pytest.approx(3999955.0)  # 4000000 + 1.5 * -30


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_point(self):
        """Distance between same point should be zero."""
        dist = haversine_distance(45.0, -122.0, 45.0, -122.0, 6378137.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_equator(self):
        """Test distance along equator (1 degree longitude)."""
        # At equator, 1 degree of longitude ≈ 111.32 km
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0, 6378137.0)
        assert dist == pytest.approx(111195.0, rel=0.01)

    def test_known_distance_meridian(self):
        """Test distance along a meridian (1 degree latitude)."""
        # 1 degree of latitude ≈ 111.32 km
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0, 6378137.0)
        assert dist == pytest.approx(111195.0, rel=0.01)

    def test_diagonal_distance(self):
        """Test distance with both lat and lon change."""
        # Portland to Seattle (approximate)
        dist = haversine_distance(45.5, -122.7, 47.6, -122.3, 6378137.0)
        # Should be roughly 235 km
        assert 230000 < dist < 240000


class TestProjectedStepDistance:
    """Tests for projected_step_distance function."""

    def test_cardinal_east(self):
        """East step (0, 1) with 10m pixels."""
        dist = projected_step_distance(0, 1, 10.0, 10.0)
        assert dist == pytest.approx(10.0)

    def test_cardinal_south(self):
        """South step (1, 0) with 10m pixels."""
        dist = projected_step_distance(1, 0, 10.0, 10.0)
        assert dist == pytest.approx(10.0)

    def test_diagonal(self):
        """Diagonal step (1, 1) with 10m pixels."""
        dist = projected_step_distance(1, 1, 10.0, 10.0)
        assert dist == pytest.approx(14.142135623730951)  # sqrt(200)

    def test_rectangular_pixels_cardinal(self):
        """Cardinal step with rectangular pixels (30m x 20m)."""
        dist = projected_step_distance(0, 1, 30.0, 20.0)
        assert dist == pytest.approx(30.0)
        dist = projected_step_distance(1, 0, 30.0, 20.0)
        assert dist == pytest.approx(20.0)

    def test_rectangular_pixels_diagonal(self):
        """Diagonal step with rectangular pixels (30m x 20m)."""
        dist = projected_step_distance(1, 1, 30.0, 20.0)
        assert dist == pytest.approx(np.sqrt(900 + 400))  # sqrt(1300)


# =============================================================================
# Unit Tests for Flow Distance Calculation Functions
# =============================================================================


class TestCalculateFlowDistanceProjected:
    """Tests for calculate_flow_distance_projected function."""

    def test_east_direction(self):
        """Direction 0 (East) should be pixel width."""
        dist = calculate_flow_distance_projected(0, 10.0, 10.0)
        assert dist == pytest.approx(10.0)

    def test_north_direction(self):
        """Direction 2 (North) should be pixel height."""
        dist = calculate_flow_distance_projected(2, 10.0, 10.0)
        assert dist == pytest.approx(10.0)

    def test_northeast_direction(self):
        """Direction 1 (NE) should be diagonal distance."""
        dist = calculate_flow_distance_projected(1, 10.0, 10.0)
        assert dist == pytest.approx(np.sqrt(200))

    def test_all_directions(self):
        """Test all 8 flow directions."""
        cardinal_dirs = [0, 2, 4, 6]  # E, N, W, S
        diagonal_dirs = [1, 3, 5, 7]  # NE, NW, SW, SE

        for d in cardinal_dirs:
            dist = calculate_flow_distance_projected(d, 10.0, 10.0)
            assert dist == pytest.approx(10.0), f"Direction {d} failed"

        for d in diagonal_dirs:
            dist = calculate_flow_distance_projected(d, 10.0, 10.0)
            assert dist == pytest.approx(np.sqrt(200)), f"Direction {d} failed"


class TestCalculateFlowDistanceGeographic:
    """Tests for calculate_flow_distance_geographic function."""

    def test_adjacent_cells_east(self):
        """Test distance between horizontally adjacent cells."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)  # 0.01 degree pixels
        dist = calculate_flow_distance_geographic(0, 0, 0, 1, gt, 6378137.0)
        # At ~46 degrees lat, 0.01 degree lon ≈ 770m
        assert 700 < dist < 850

    def test_adjacent_cells_south(self):
        """Test distance between vertically adjacent cells."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)
        dist = calculate_flow_distance_geographic(0, 0, 1, 0, gt, 6378137.0)
        # 0.01 degree lat ≈ 1111m
        assert 1000 < dist < 1200

    def test_same_cell(self):
        """Distance between same cell should be zero."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)
        dist = calculate_flow_distance_geographic(5, 5, 5, 5, gt, 6378137.0)
        assert dist == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Unit Tests for is_geographic Helper
# =============================================================================


class TestIsGeographic:
    """Tests for is_geographic function."""

    def test_projected_crs(self):
        """UTM zone should not be geographic."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32610)  # UTM Zone 10N
        assert is_geographic(srs) is False

    def test_geographic_crs_wgs84(self):
        """WGS84 should be geographic."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        assert is_geographic(srs) is True

    def test_geographic_crs_nad83(self):
        """NAD83 should be geographic."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4269)  # NAD83
        assert is_geographic(srs) is True


# =============================================================================
# Unit Tests for trace_path_from_cell
# =============================================================================


class TestTracePathFromCell:
    """Tests for trace_path_from_cell function."""

    def test_simple_linear_path(self):
        """Test tracing a simple linear flow path going east."""
        # All cells flow east (direction 0)
        fdr = np.array(
            [
                [0, 0, 0, 0, 9],  # 9 = outlet
            ],
            dtype=np.ubyte,
        )
        path = trace_path_from_cell(fdr, 0, 0, (0, 4))
        assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_path_to_self(self):
        """Starting at drainage point should return single-cell path."""
        fdr = np.array([[0, 0, 0]], dtype=np.ubyte)
        path = trace_path_from_cell(fdr, 0, 1, (0, 1))
        assert path == [(0, 1)]

    def test_zigzag_path(self):
        """Test tracing a zig-zag path."""
        # Flow: (0,0)->E->(0,1)->S->(1,1)->E->(1,2)
        fdr = np.array(
            [
                [0, 6, 9],  # 0=E, 6=S
                [9, 0, 9],  # 0=E
            ],
            dtype=np.ubyte,
        )
        path = trace_path_from_cell(fdr, 0, 0, (1, 2))
        assert path == [(0, 0), (0, 1), (1, 1), (1, 2)]

    def test_stops_at_nodata(self):
        """Path should stop at nodata values."""
        fdr = np.array(
            [
                [0, 0, FLOW_DIRECTION_NODATA, 0],
            ],
            dtype=np.ubyte,
        )
        path = trace_path_from_cell(fdr, 0, 0, (0, 3))
        # Should stop at cell (0,2) with nodata
        assert path == [(0, 0), (0, 1), (0, 2)]

    def test_stops_at_boundary(self):
        """Path should stop at raster boundary."""
        fdr = np.array(
            [
                [2, 2, 2],  # All flow north (out of bounds)
            ],
            dtype=np.ubyte,
        )
        path = trace_path_from_cell(fdr, 0, 0, (0, 2))
        # Should stop immediately since north is out of bounds
        assert path == [(0, 0)]


# =============================================================================
# Unit Tests for Path Distance Calculations
# =============================================================================


class TestCalculatePathDistanceProjected:
    """Tests for calculate_path_distance_projected function."""

    def test_single_cell_path(self):
        """Single cell path should have zero distance."""
        path = [(0, 0)]
        dist = calculate_path_distance_projected(path, 10.0, 10.0)
        assert dist == pytest.approx(0.0)

    def test_linear_horizontal_path(self):
        """Horizontal path of 3 cells should be 2 * pixel_width."""
        path = [(0, 0), (0, 1), (0, 2)]
        dist = calculate_path_distance_projected(path, 10.0, 10.0)
        assert dist == pytest.approx(20.0)

    def test_linear_vertical_path(self):
        """Vertical path of 3 cells should be 2 * pixel_height."""
        path = [(0, 0), (1, 0), (2, 0)]
        dist = calculate_path_distance_projected(path, 10.0, 10.0)
        assert dist == pytest.approx(20.0)

    def test_diagonal_path(self):
        """Diagonal path should use diagonal distances."""
        path = [(0, 0), (1, 1), (2, 2)]
        dist = calculate_path_distance_projected(path, 10.0, 10.0)
        expected = 2 * np.sqrt(200)  # Two diagonal steps
        assert dist == pytest.approx(expected)

    def test_mixed_path(self):
        """Mixed cardinal and diagonal path."""
        path = [(0, 0), (0, 1), (1, 2)]  # East then SE
        dist = calculate_path_distance_projected(path, 10.0, 10.0)
        expected = 10.0 + np.sqrt(200)  # One cardinal + one diagonal
        assert dist == pytest.approx(expected)


class TestCalculatePathDistanceGeographic:
    """Tests for calculate_path_distance_geographic function."""

    def test_single_cell_path(self):
        """Single cell path should have zero distance."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)
        path = [(0, 0)]
        dist = calculate_path_distance_geographic(path, gt, 6378137.0)
        assert dist == pytest.approx(0.0)

    def test_two_cell_horizontal_path(self):
        """Two cell horizontal path distance."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)
        path = [(0, 0), (0, 1)]
        dist = calculate_path_distance_geographic(path, gt, 6378137.0)
        # Should be around 700-850m at 46 degrees lat
        assert 700 < dist < 850

    def test_multi_cell_path(self):
        """Multi-cell path should accumulate distances."""
        gt = (-122.0, 0.01, 0.0, 46.0, 0.0, -0.01)
        path = [(0, 0), (0, 1), (0, 2)]
        dist = calculate_path_distance_geographic(path, gt, 6378137.0)
        # Two steps should be roughly double one step
        single_step = calculate_path_distance_geographic(
            [(0, 0), (0, 1)], gt, 6378137.0
        )
        assert dist == pytest.approx(2 * single_step, rel=0.01)


# =============================================================================
# Unit Tests for find_all_upstream_basins
# =============================================================================


def create_basin_graph(graph_dict: dict) -> Dict:
    """Create a typed basin_graph dict from a plain Python dict."""
    # Type: Dict[int64, List[int64]]
    list_type = types.ListType(int64)
    basin_graph = Dict.empty(int64, list_type)
    for key, values in graph_dict.items():
        lst = List.empty_list(int64)
        for v in values:
            lst.append(v)
        basin_graph[key] = lst
    return basin_graph


def create_basin_max_cells(cells_dict: dict) -> Dict:
    """Create a typed basin_max_cells dict from a plain Python dict."""
    # Type: Dict[int64, Tuple[float64, float64, float64]]
    tuple_type = types.UniTuple(types.float64, 3)
    basin_max_cells = Dict.empty(int64, tuple_type)
    for key, value in cells_dict.items():
        basin_max_cells[key] = value
    return basin_max_cells


def create_drainage_points(points_dict: dict) -> Dict:
    """Create a typed drainage_points dict from a plain Python dict."""
    # Type: Dict[Tuple[int64, int64], int64]
    key_type = types.UniTuple(int64, 2)
    drainage_points = Dict.empty(key_type, int64)
    for key, value in points_dict.items():
        drainage_points[key] = value
    return drainage_points


class TestFindAllUpstreamBasins:
    """Tests for find_all_upstream_basins function."""

    def test_no_upstream_basins(self):
        """Basin with no upstream basins should return only itself."""
        basin_graph = create_basin_graph({1: []})
        result = find_all_upstream_basins(1, basin_graph)
        assert list(result) == [1]

    def test_single_upstream_basin(self):
        """Basin with one upstream basin."""
        basin_graph = create_basin_graph({1: [2], 2: []})
        result = find_all_upstream_basins(1, basin_graph)
        assert set(result) == {1, 2}

    def test_chain_of_upstream_basins(self):
        """Chain: 3 -> 2 -> 1 (downstream)."""
        basin_graph = create_basin_graph({1: [2], 2: [3], 3: []})
        result = find_all_upstream_basins(1, basin_graph)
        assert set(result) == {1, 2, 3}

    def test_branching_upstream_basins(self):
        """Multiple basins flow into one: 2,3 -> 1."""
        basin_graph = create_basin_graph({1: [2, 3], 2: [], 3: []})
        result = find_all_upstream_basins(1, basin_graph)
        assert set(result) == {1, 2, 3}

    def test_basin_not_in_graph(self):
        """Basin not in graph should return only itself."""
        basin_graph = create_basin_graph({1: []})
        result = find_all_upstream_basins(99, basin_graph)
        assert list(result) == [99]


# =============================================================================
# Unit Tests for calculate_upstream_flow_length
# =============================================================================


class TestCalculateUpstreamFlowLength:
    """Tests for calculate_upstream_flow_length function."""

    def test_simple_linear_drainage(self):
        """Simple linear drainage with single outlet."""
        # All flow east to outlet at (0, 4)
        fdr = np.array(
            [
                [0, 0, 0, 0, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_points = create_drainage_points({(0, 4): 1})
        flow_length, basin_labels, _, basin_max_cells = calculate_upstream_flow_length(
            fdr,
            drainage_points,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
            is_geographic=False,
            geotransform=(0.0, 10.0, 0.0, 10.0, 0.0, -10.0),
            semi_major=6378137.0,
        )

        # All cells should be labeled with basin 1
        assert np.all(basin_labels == 1)

        # Flow lengths should increase from outlet
        assert flow_length[0, 4] == pytest.approx(0.0)  # Outlet
        assert flow_length[0, 3] == pytest.approx(10.0)
        assert flow_length[0, 2] == pytest.approx(20.0)
        assert flow_length[0, 1] == pytest.approx(30.0)
        assert flow_length[0, 0] == pytest.approx(40.0)

        # Max cell should be at (0, 0)
        assert basin_max_cells[1] == (0.0, 0.0, 40.0)

    def test_two_separate_basins(self):
        """Two independent basins with separate outlets."""
        # Left flows west, right flows east
        fdr = np.array(
            [
                [9, 4, 4, 0, 0, 9],  # Outlets at (0,0) and (0,5)
            ],
            dtype=np.ubyte,
        )
        drainage_points = create_drainage_points({(0, 0): 1, (0, 5): 2})
        flow_length, basin_labels, _, _ = calculate_upstream_flow_length(
            fdr,
            drainage_points,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
            is_geographic=False,
            geotransform=(0.0, 10.0, 0.0, 10.0, 0.0, -10.0),
            semi_major=6378137.0,
        )

        # Check basin labels
        assert basin_labels[0, 0] == 1
        assert basin_labels[0, 1] == 1
        assert basin_labels[0, 2] == 1
        assert basin_labels[0, 3] == 2
        assert basin_labels[0, 4] == 2
        assert basin_labels[0, 5] == 2

        # Check flow lengths
        assert flow_length[0, 0] == pytest.approx(0.0)
        assert flow_length[0, 2] == pytest.approx(20.0)
        assert flow_length[0, 5] == pytest.approx(0.0)
        assert flow_length[0, 3] == pytest.approx(20.0)

    def test_diagonal_flow(self):
        """Test diagonal flow distance calculation."""
        # All flow southeast to outlet at (2, 2)
        fdr = np.array(
            [
                [7, 7, 7],
                [7, 7, 7],
                [7, 7, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_points = create_drainage_points({(2, 2): 1})
        flow_length, _, _, _ = calculate_upstream_flow_length(
            fdr,
            drainage_points,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
            is_geographic=False,
            geotransform=(0.0, 10.0, 0.0, 30.0, 0.0, -10.0),
            semi_major=6378137.0,
        )

        diag_dist = np.sqrt(200)
        assert flow_length[2, 2] == pytest.approx(0.0)
        assert flow_length[1, 1] == pytest.approx(diag_dist)
        assert flow_length[0, 0] == pytest.approx(2 * diag_dist)


# =============================================================================
# Unit Tests for trace_longest_flow_path
# =============================================================================


class TestTraceLongestFlowPath:
    """Tests for trace_longest_flow_path function."""

    def test_single_basin_longest_path(self):
        """Test tracing longest path in single basin."""
        # Linear flow east
        fdr = np.array(
            [
                [0, 0, 0, 0, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_point = (0, 4)
        dp_id = 1

        # Create basin_graph and basin_max_cells
        basin_graph = create_basin_graph({1: []})
        basin_max_cells = create_basin_max_cells({1: (0.0, 0.0, 40.0)})

        path = trace_longest_flow_path(
            fdr,
            drainage_point,
            dp_id,
            basin_graph,
            basin_max_cells,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
            is_geographic=False,
            geotransform=(0.0, 10.0, 0.0, 10.0, 0.0, -10.0),
            semi_major=6378137.0,
        )

        assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_path_with_upstream_basin(self):
        """Test that path considers upstream basins."""
        # Two basins: 2 flows into 1
        fdr = np.array(
            [
                [0, 0, 0, 0, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_point = (0, 4)
        dp_id = 1

        # Basin 2 is upstream of basin 1, with max cell at (0,0)
        basin_graph = create_basin_graph({1: [2], 2: []})

        # Basin 1's max is at (0,2), basin 2's max is at (0,0)
        basin_max_cells = create_basin_max_cells(
            {
                1: (0.0, 2.0, 20.0),
                2: (0.0, 0.0, 40.0),
            }
        )

        path = trace_longest_flow_path(
            fdr,
            drainage_point,
            dp_id,
            basin_graph,
            basin_max_cells,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
            is_geographic=False,
            geotransform=(0.0, 10.0, 0.0, 10.0, 0.0, -10.0),
            semi_major=6378137.0,
        )

        # Should trace from (0,0) which is farther
        assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


# =============================================================================
# Unit Tests for create_longest_flow_path_vectors
# =============================================================================


class TestCreateLongestFlowPathVectors:
    """Tests for create_longest_flow_path_vectors function."""

    def test_creates_valid_geopackage(self):
        """Test that function creates a valid GeoPackage with correct features."""
        fdr = np.array(
            [
                [0, 0, 0, 0, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_points = {(0, 4): 1}
        gt = (0.0, 10.0, 0.0, 10.0, 0.0, -10.0)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32610)
        projection = srs.ExportToWkt()

        basin_graph = create_basin_graph({1: []})
        basin_max_cells = create_basin_max_cells({1: (0.0, 0.0, 40.0)})

        output_file = "/vsimem/test_vectors.gpkg"

        create_longest_flow_path_vectors(
            fdr,
            drainage_points,
            gt,
            projection,
            output_file,
            is_geographic=False,
            srs_input=srs,
            basin_graph=basin_graph,
            basin_max_cells=basin_max_cells,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
        )

        # Read and verify output
        ds = ogr.Open(output_file)
        assert ds is not None
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 1

        feature = layer.GetNextFeature()
        assert feature.GetField("dp_id") == 1
        assert feature.GetField("length_m") == pytest.approx(40.0)
        assert feature.GetField("length_ft") == pytest.approx(40.0 * 3.28084)

        geom = feature.GetGeometryRef()
        assert geom.GetPointCount() == 5

        ds = None
        gdal.Unlink(output_file)

    def test_multiple_drainage_points(self):
        """Test with multiple drainage points."""
        fdr = np.array(
            [
                [9, 4, 4, 0, 0, 9],
            ],
            dtype=np.ubyte,
        )
        drainage_points = {(0, 0): 1, (0, 5): 2}
        gt = (0.0, 10.0, 0.0, 10.0, 0.0, -10.0)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32610)
        projection = srs.ExportToWkt()

        basin_graph = create_basin_graph({1: [], 2: []})
        basin_max_cells = create_basin_max_cells(
            {
                1: (0.0, 2.0, 20.0),
                2: (0.0, 3.0, 20.0),
            }
        )

        output_file = "/vsimem/test_multi_vectors.gpkg"

        create_longest_flow_path_vectors(
            fdr,
            drainage_points,
            gt,
            projection,
            output_file,
            is_geographic=False,
            srs_input=srs,
            basin_graph=basin_graph,
            basin_max_cells=basin_max_cells,
            pixel_size_x=10.0,
            pixel_size_y=10.0,
        )

        ds = ogr.Open(output_file)
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 2

        ds = None
        gdal.Unlink(output_file)
