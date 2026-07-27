import contextlib
import uuid

import numpy as np
import pytest
from osgeo import gdal

from overflow import accumulation
from overflow._flow_accumulation.core import (
    single_tile_flow_accumulation,
    single_tile_flow_accumulation_weighted,
)
from overflow._flow_accumulation.tiled import (
    _flow_accumulation_tiled,
    _flow_accumulation_weighted_tiled,
)
from overflow._util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
    FLOW_EXTERNAL,
)

# Fixtures


@pytest.fixture(name="fdr_arrays")
def fixture_fdr_arrays():
    """Fixture providing multiple flow direction arrays for testing."""
    fdr1 = np.array(
        [
            [2, 2, 2, 2, 2, 2, 2],
            [1, 2, 3, 1, 1, 2, 3],
            [2, 2, 2, 1, 2, 2, 3],
            [2, 2, 3, 4, 2, 3, 3],
            [4, 2, 2, 3, 3, 4, 4],
            [3, 3, 7, 0, 2, 3, 4],
            [3, 7, 0, 1, 2, 2, 4],
        ],
        dtype=np.ubyte,
    )

    fdr2 = np.array(
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

    fdr3 = np.array(
        [
            [5, 5, 6, 5, 5, 6],
            [5, 6, 5, 5, 6, 5],
            [6, 5, 5, 6, 5, 6],
            [5, 5, 6, 5, 6, 5],
            [5, 6, 5, 6, 5, 5],
            [6, 5, 6, 5, 5, 5],
        ],
        dtype=np.ubyte,
    )

    return {"fdr1": fdr1, "fdr2": fdr2, "fdr3": fdr3}


@pytest.fixture(name="fdr_file_paths")
def fixture_fdr_file_paths(fdr_arrays):
    """Fixture creating temporary file paths for flow direction arrays."""
    file_paths = {}
    for name, fdr in fdr_arrays.items():
        output_path = f"/vsimem/test_{name}.tif"
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            output_path, fdr.shape[1], fdr.shape[0], 1, gdal.GDT_Byte
        )
        band = dataset.GetRasterBand(1)
        band.WriteArray(fdr)
        band.SetNoDataValue(FLOW_DIRECTION_NODATA)
        dataset.FlushCache()
        dataset = None
        file_paths[name] = output_path

    yield file_paths

    for path in file_paths.values():
        gdal.Unlink(path)


@pytest.fixture(name="weights_arrays")
def fixture_weights_arrays(fdr_arrays):
    """Fixture providing a deterministic, nodata-free weights raster for each
    flow direction array, matching its shape."""
    weights = {}
    for name, fdr in fdr_arrays.items():
        weights[name] = (np.arange(fdr.size).reshape(fdr.shape) % 5 + 1).astype(
            np.float64
        )
    return weights


@pytest.fixture(name="weights_file_paths")
def fixture_weights_file_paths(weights_arrays):
    """Fixture creating temporary file paths for weights arrays."""
    file_paths = {}
    for name, weights in weights_arrays.items():
        output_path = f"/vsimem/test_weights_{name}.tif"
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            output_path, weights.shape[1], weights.shape[0], 1, gdal.GDT_Float64
        )
        band = dataset.GetRasterBand(1)
        band.WriteArray(weights)
        band.SetNoDataValue(-9999.0)
        dataset.FlushCache()
        dataset = None
        file_paths[name] = output_path

    yield file_paths

    for path in file_paths.values():
        gdal.Unlink(path)


# Helper functions


def generate_unique_filepath():
    """Generate a unique filepath for temporary output."""
    unique_id = uuid.uuid4()
    return f"/vsimem/test_output_{unique_id}.tif"


@contextlib.contextmanager
def temporary_dataset():
    """
    Create a temporary output dataset for flow accumulation.

    This context manager creates a unique filename for each test,
    sets up the dataset, and ensures it's properly cleaned up after use.

    Args:
        shape (tuple): The shape of the output dataset (rows, cols).

    Yields:
        str: The path to the temporary output dataset.
    """
    output_path = generate_unique_filepath()

    yield output_path
    try:
        gdal.Unlink(output_path)
    except RuntimeError as exc:
        print(f"Failed to clean up temporary dataset: {exc}")


def read_output_dataset(path):
    """Read the output dataset and return as numpy array."""
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    fac = band.ReadAsArray()
    gdal.Unlink(path)
    return fac


# Tests


@pytest.mark.parametrize(
    "fdr_key, expected_fac",
    [
        (
            "fdr1",
            np.array(
                [
                    [1, 27, 1, 1, 2, 11, 1],
                    [3, 21, 2, 1, 5, 4, 1],
                    [2, 20, 1, 1, 3, 2, 1],
                    [1, 2, 17, 14, 1, 1, 1],
                    [2, 1, 1, 1, 13, 2, 1],
                    [1, 1, 1, 1, 6, 4, 1],
                    [1, 1, 1, 3, 1, 2, 1],
                ],
                dtype=np.int64,
            ),
        ),
        (
            "fdr2",
            np.array(
                [
                    [1, 3, 2, 1, 1, 1],
                    [8, 7, 3, 2, 2, 1],
                    [1, 7, 6, 5, 2, 1],
                    [9, 1, 5, 4, 3, 1],
                    [1, 7, 1, 2, 1, 1],
                    [9, 1, 4, 1, 3, 2],
                ],
                dtype=np.int64,
            ),
        ),
        (
            "fdr3",
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [2, 1, 3, 2, 1, 2],
                    [1, 5, 3, 1, 4, 1],
                    [7, 4, 1, 6, 1, 2],
                    [5, 1, 8, 1, 4, 1],
                    [1, 10, 1, 6, 2, 1],
                ],
                dtype=np.int64,
            ),
        ),
    ],
)
def test_single_tile_flow_accumulation(fdr_arrays, fdr_key, expected_fac):
    """Test single tile flow accumulation for different flow direction arrays."""
    fac, _ = single_tile_flow_accumulation(fdr_arrays[fdr_key])
    np.testing.assert_array_equal(fac, expected_fac)


def test_single_tile_flow_accumulation_with_nodata(fdr_arrays):
    """Test single tile flow accumulation with nodata values."""
    fdr = fdr_arrays["fdr1"].copy()
    fdr[0, 1] = FLOW_DIRECTION_NODATA
    fac, _ = single_tile_flow_accumulation(fdr)
    expected_fac = np.array(
        [
            [1, FLOW_ACCUMULATION_NODATA, 1, 1, 2, 11, 1],
            [3, 21, 2, 1, 5, 4, 1],
            [2, 20, 1, 1, 3, 2, 1],
            [1, 2, 17, 14, 1, 1, 1],
            [2, 1, 1, 1, 13, 2, 1],
            [1, 1, 1, 1, 6, 4, 1],
            [1, 1, 1, 3, 1, 2, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(fac, expected_fac)


def test_links_in_single_tile_flow_accumulation(fdr_arrays):
    """Test the links output in single tile flow accumulation."""
    _, links = single_tile_flow_accumulation(fdr_arrays["fdr1"])
    # Test perimeter cells in links
    assert np.all(links[0, :] == FLOW_EXTERNAL)
    assert np.all(links[1:4, 0] == (0, 1))
    assert np.all(links[4:, 0] == FLOW_EXTERNAL)
    assert np.all(links[1:4, 6] == (0, 5))
    assert np.all(links[4:, 6] == (0, 1))
    assert np.all(links[6, 1] == FLOW_EXTERNAL)
    assert np.all(links[6, 2:6] == (0, 1))


@pytest.mark.parametrize("tile_size", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("fdr_key", ["fdr1", "fdr2", "fdr3"])
def test_tiled_flow_accumulation(fdr_file_paths, fdr_arrays, tile_size, fdr_key):
    """Test tiled flow accumulation for different tile sizes and flow direction arrays."""
    fdr_path = fdr_file_paths[fdr_key]
    fdr = fdr_arrays[fdr_key]

    with temporary_dataset() as output_path:
        _flow_accumulation_tiled(fdr_path, output_path, tile_size)
        fac = read_output_dataset(output_path)

        expected_fac, _ = single_tile_flow_accumulation(fdr)
        np.testing.assert_array_equal(fac, expected_fac)


def test_tiled_flow_accumulation_with_nodata(fdr_file_paths):
    """Test tiled flow accumulation with nodata values."""
    fdr_path = fdr_file_paths["fdr1"]
    fdr_ds = gdal.Open(fdr_path, gdal.GA_Update)
    band = fdr_ds.GetRasterBand(1)
    fdr = band.ReadAsArray()
    fdr[0, 1] = FLOW_DIRECTION_NODATA
    band.WriteArray(fdr)
    fdr_ds = None

    with temporary_dataset() as output_path:
        _flow_accumulation_tiled(fdr_path, output_path, 4)
        fac = read_output_dataset(output_path)

        expected_fac, _ = single_tile_flow_accumulation(fdr)
        np.testing.assert_array_equal(fac, expected_fac)


def test_tiled_flow_accumulation_with_cycle(capfd):
    """A flow direction raster containing a two-cell cycle (an invalid input
    that upstream bugs can produce) must not corrupt cells outside the cycle,
    must not inflate the cycle cells by repeatedly applying global offsets,
    and must print a warning with tile and cell coordinates."""
    # every cell flows east off the raster, except (1, 5) which points west,
    # forming a cycle with (1, 4); with chunk 4 the pair sits inside tile
    # (0, 1) and receives inflow 4 across the tile seam from row 1 of tile
    # (0, 0)
    fdr = np.zeros((8, 8), dtype=np.ubyte)
    fdr[1, 5] = 4
    fdr_path = generate_unique_filepath()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fdr_path, 8, 8, 1, gdal.GDT_Byte)
    band = dataset.GetRasterBand(1)
    band.WriteArray(fdr)
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    dataset.FlushCache()
    dataset = None

    with temporary_dataset() as output_path:
        _flow_accumulation_tiled(fdr_path, output_path, 4)
        fac = read_output_dataset(output_path)
    gdal.Unlink(fdr_path)

    expected = np.tile(np.arange(1, 9, dtype=np.int64), (8, 1))
    # the cycle swallows row 1: its cells get the upstream inflow exactly
    # once, nothing propagates past the pair, and cells east of the pair
    # restart at 1
    expected[1] = [1, 2, 3, 4, 4, 4, 1, 2]
    np.testing.assert_array_equal(fac, expected)

    captured = capfd.readouterr()
    assert "Cycle detected" in captured.out
    assert "tile:" in captured.out


# Weighted accumulation tests


def _weighted_chain_fdr():
    """A simple single-row chain, every cell flowing east off the tile:
    cell0 -> cell1 -> cell2 -> cell3 -> cell4 -> (off tile)."""
    return np.zeros((1, 5), dtype=np.ubyte)


@pytest.mark.parametrize(
    "fdr_key",
    ["fdr1", "fdr2", "fdr3"],
)
def test_single_tile_flow_accumulation_weighted_matches_unweighted_with_unit_weights(
    fdr_arrays, fdr_key
):
    """With every weight equal to 1, weighted accumulation must equal the
    unweighted cell-count accumulation (cast to float)."""
    fdr = fdr_arrays[fdr_key]
    weights = np.ones_like(fdr, dtype=np.float64)
    expected_fac, _ = single_tile_flow_accumulation(fdr)
    fac, _ = single_tile_flow_accumulation_weighted(
        fdr, weights, -9999.0, 0, create_links=False
    )
    np.testing.assert_array_equal(fac, expected_fac.astype(np.float64))


@pytest.mark.parametrize(
    "fdr_key",
    ["fdr1", "fdr2", "fdr3"],
)
def test_single_tile_flow_accumulation_weighted_scales_with_constant_weight(
    fdr_arrays, fdr_key
):
    """With every weight equal to a constant c, weighted accumulation at each
    cell must equal c times the unweighted cell count at that cell."""
    fdr = fdr_arrays[fdr_key]
    weights = np.full_like(fdr, 3, dtype=np.float64)
    expected_fac, _ = single_tile_flow_accumulation(fdr)
    fac, _ = single_tile_flow_accumulation_weighted(
        fdr, weights, -9999.0, 0, create_links=False
    )
    np.testing.assert_allclose(fac, 3.0 * expected_fac.astype(np.float64))


def test_single_tile_flow_accumulation_weighted_zero_nodata():
    """A nodata weight cell in "zero" mode contributes 0 but routing
    continues normally: downstream cells still receive the (unaffected)
    upstream accumulation."""
    fdr = _weighted_chain_fdr()
    weights = np.array([[1.0, 1.0, -9999.0, 1.0, 1.0]])
    fac, _ = single_tile_flow_accumulation_weighted(
        fdr, weights, -9999.0, 0, create_links=False
    )
    expected = np.array([[1.0, 2.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(fac, expected)


def test_single_tile_flow_accumulation_weighted_propagate_nodata():
    """A nodata weight cell in "propagate" mode poisons its own accumulation
    and everything downstream of it with NaN, while leaving cells upstream of
    it unaffected."""
    fdr = _weighted_chain_fdr()
    weights = np.array([[1.0, 1.0, -9999.0, 1.0, 1.0]])
    fac, _ = single_tile_flow_accumulation_weighted(
        fdr, weights, -9999.0, 1, create_links=False
    )
    expected = np.array([[1.0, 2.0, np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(fac, expected)


@pytest.mark.parametrize("tile_size", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("fdr_key", ["fdr1", "fdr2", "fdr3"])
def test_tiled_flow_accumulation_weighted(
    fdr_file_paths, fdr_arrays, weights_arrays, weights_file_paths, tile_size, fdr_key
):
    """Tiled weighted flow accumulation must match the single-tile (core)
    reference result for every tile size, with no nodata weight cells."""
    fdr_path = fdr_file_paths[fdr_key]
    weights_path = weights_file_paths[fdr_key]

    with temporary_dataset() as output_path:
        _flow_accumulation_weighted_tiled(
            fdr_path, weights_path, output_path, tile_size
        )
        fac = read_output_dataset(output_path)

        expected_fac, _ = single_tile_flow_accumulation_weighted(
            fdr_arrays[fdr_key],
            weights_arrays[fdr_key],
            -9999.0,
            0,
            create_links=False,
        )
        np.testing.assert_allclose(fac, expected_fac)


@pytest.mark.parametrize("weights_nodata_mode", ["zero", "propagate"])
@pytest.mark.parametrize("tile_size", [2, 3, 4])
def test_tiled_flow_accumulation_weighted_with_nodata(
    fdr_file_paths, fdr_arrays, tile_size, weights_nodata_mode
):
    """Tiled weighted accumulation with a nodata weight cell must match the
    single-tile reference for both nodata modes, across tile sizes, proving
    the cross-tile global-offset propagation (including NaN self-propagation
    in "propagate" mode) matches the single-tile algorithm."""
    fdr = fdr_arrays["fdr1"]
    weights = np.ones_like(fdr, dtype=np.float64)
    weights[2, 3] = -9999.0
    mode = 0 if weights_nodata_mode == "zero" else 1

    fdr_path = "/vsimem/test_weighted_nodata_fdr.tif"
    driver = gdal.GetDriverByName("GTiff")
    fdr_ds = driver.Create(fdr_path, fdr.shape[1], fdr.shape[0], 1, gdal.GDT_Byte)
    fdr_band = fdr_ds.GetRasterBand(1)
    fdr_band.WriteArray(fdr)
    fdr_band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    fdr_ds.FlushCache()
    fdr_ds = None

    weights_path = "/vsimem/test_weighted_nodata_weights.tif"
    weights_ds = driver.Create(
        weights_path, weights.shape[1], weights.shape[0], 1, gdal.GDT_Float64
    )
    weights_band = weights_ds.GetRasterBand(1)
    weights_band.WriteArray(weights)
    weights_band.SetNoDataValue(-9999.0)
    weights_ds.FlushCache()
    weights_ds = None

    try:
        with temporary_dataset() as output_path:
            _flow_accumulation_weighted_tiled(
                fdr_path,
                weights_path,
                output_path,
                tile_size,
                weights_nodata_mode=weights_nodata_mode,
            )
            fac = read_output_dataset(output_path)

            expected_fac, _ = single_tile_flow_accumulation_weighted(
                fdr, weights, -9999.0, mode, create_links=False
            )
            np.testing.assert_allclose(fac, expected_fac, equal_nan=True)
    finally:
        gdal.Unlink(fdr_path)
        gdal.Unlink(weights_path)


def test_accumulation_weights_shape_mismatch_raises(fdr_file_paths):
    """accumulation() must raise ValueError when the weights raster's shape
    does not match the flow direction raster's shape."""
    fdr_path = fdr_file_paths["fdr1"]  # 7x7
    mismatched_weights = np.ones((6, 6), dtype=np.float64)
    weights_path = "/vsimem/test_mismatched_weights.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(weights_path, 6, 6, 1, gdal.GDT_Float64)
    band = dataset.GetRasterBand(1)
    band.WriteArray(mismatched_weights)
    band.SetNoDataValue(-9999.0)
    dataset.FlushCache()
    dataset = None

    try:
        with temporary_dataset() as output_path, pytest.raises(ValueError):
            accumulation(fdr_path, output_path, chunk_size=0, weights_path=weights_path)
    finally:
        gdal.Unlink(weights_path)


def test_accumulation_invalid_weights_nodata_mode_raises(
    fdr_file_paths, weights_file_paths
):
    """accumulation() must raise ValueError for an unrecognized
    weights_nodata_mode."""
    fdr_path = fdr_file_paths["fdr1"]
    weights_path = weights_file_paths["fdr1"]
    with temporary_dataset() as output_path, pytest.raises(ValueError):
        accumulation(
            fdr_path,
            output_path,
            chunk_size=0,
            weights_path=weights_path,
            weights_nodata_mode="bogus",
        )
