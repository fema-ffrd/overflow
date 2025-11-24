import contextlib
import uuid

import numpy as np
import pytest
from osgeo import gdal

from overflow._flow_accumulation.core import single_tile_flow_accumulation
from overflow._flow_accumulation.tiled import _flow_accumulation_tiled
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
