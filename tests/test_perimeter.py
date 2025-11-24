import numpy as np
import pytest

from overflow._util.perimeter import Int64Perimeter, get_tile_perimeter
from overflow._util.raster import Corner, Side


@pytest.fixture(name="test_edge_cells")
def fixture_test_edge_cells():
    """Create a sample TileEdgeCells for testing."""
    array = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [15, -1, -1, -1, -1, 6],
            [14, -1, -1, -1, -1, 7],
            [13, 12, 11, 10, 9, 8],
        ],
        dtype=np.int64,
    )
    perimeter_data = get_tile_perimeter(array)
    return Int64Perimeter(perimeter_data, 4, 6, 0)


@pytest.fixture(name="test_perimeter_indices")
def fixture_test_perimeter_indices():
    """The row and column indices of the perimeter cells. In order
    of the perimeter flattened indices."""
    return [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (3, 4),
        (3, 3),
        (3, 2),
        (3, 1),
        (3, 0),
        (2, 0),
        (1, 0),
    ]


def test_get_index(test_edge_cells, test_perimeter_indices):
    """Test the get_flattened_index method."""
    # check that the flattend indec for all perimeter cells are correct
    flattened_indices = [
        test_edge_cells.get_index(row, col) for row, col in test_perimeter_indices
    ]
    assert flattened_indices == list(range(test_edge_cells.size()))
    assert test_edge_cells.get_index_corner(Corner.TOP_LEFT) == 0
    assert test_edge_cells.get_index_corner(Corner.TOP_RIGHT) == 5
    assert test_edge_cells.get_index_corner(Corner.BOTTOM_RIGHT) == 8
    assert test_edge_cells.get_index_corner(Corner.BOTTOM_LEFT) == 13
    top_indices = [test_edge_cells.get_index_side(Side.TOP, i) for i in range(6)]
    assert top_indices == [0, 1, 2, 3, 4, 5]
    right_indices = [test_edge_cells.get_index_side(Side.RIGHT, i) for i in range(4)]
    assert right_indices == [5, 6, 7, 8]
    bottom_indices = [test_edge_cells.get_index_side(Side.BOTTOM, i) for i in range(6)]
    assert bottom_indices == [13, 12, 11, 10, 9, 8]
    left_indices = [test_edge_cells.get_index_side(Side.LEFT, i) for i in range(4)]
    assert left_indices == [0, 15, 14, 13]


def test_get_row_col(test_edge_cells, test_perimeter_indices):
    """Test the get_row_col method."""
    # check that the row and col for all flattened indices are correct
    row_col_indices = [
        test_edge_cells.get_row_col(flattened_index)
        for flattened_index in range(test_edge_cells.size())
    ]
    assert row_col_indices == test_perimeter_indices


def test_size(test_edge_cells):
    """Test the size property."""
    assert test_edge_cells.size() == 16


def test_get_side(test_edge_cells):
    """Test the get_side method."""
    # check that the sides are correct
    sides = [
        list(test_edge_cells.get_side(side))
        for side in [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]
    ]
    assert sides == [
        [0, 1, 2, 3, 4, 5],
        [5, 6, 7, 8],
        [13, 12, 11, 10, 9, 8],
        [0, 15, 14, 13],
    ]
