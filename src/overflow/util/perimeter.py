import numpy as np
from numba import float32, int64, njit, uint8  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.types import ListType

from overflow.util.raster import Corner, Side


@njit
def get_tile_perimeter(array: np.ndarray) -> np.ndarray:
    rows = len(array)
    cols = len(array[0])
    perimeter = np.empty(2 * (rows + cols) - 4, dtype=array.dtype)
    # Fill the perimeter array
    perimeter[:cols] = array[0]  # top
    perimeter[cols : cols + rows - 2] = array[1:-1, -1]  # right
    perimeter[cols + rows - 2 : 2 * cols + rows - 2] = array[-1, ::-1]  # bottom
    perimeter[2 * cols + rows - 2 :] = array[-2:0:-1, 0]  # left
    return perimeter


def create_perimeter_class(numba_type):
    # numba doesn't have templates yet, so this is a workaround
    # to not have to write the same code for each data type
    spec = [("data", numba_type[:])]

    @jitclass(spec)
    class Perimeter:
        """
        A class representing the cells on the perimeter of a 2D array (tile).

        The perimeter cells are stored in a flattened 1D array in a clockwise order
        starting from the top-left corner.

        Attributes:
            rows (int): The number of rows in the original 2D array.
            cols (int): The number of columns in the original 2D array.
            index_offset (int): The offset to add to the flattened index to get the index in the global array
            perimeter (np.ndarray): A 1D contiguous array containing the flattened perimeter cells.
        """

        rows: int  # type: ignore[annotation-unchecked]
        cols: int  # type: ignore[annotation-unchecked]
        index_offset: int  # type: ignore[annotation-unchecked]
        data: np.ndarray  # type: ignore[annotation-unchecked]

        def __init__(self, data: np.ndarray, rows: int, cols: int, tile_index: int):
            self.rows = rows
            self.cols = cols
            self.data = data
            self.index_offset = tile_index * self.data.size

        def get_index(self, row, col):
            """
            Returns the flattened index of the cell at the given (row, col) coordinates.
            """
            if row == 0:
                return col
            elif row == self.rows - 1:
                return self.cols + self.rows - 2 + self.cols - col - 1
            elif col == self.cols - 1:
                return self.cols + row - 1
            else:
                return 2 * self.cols + self.rows - 2 + self.rows - row - 2

        def get_index_corner(self, corner: Corner):
            """
            Returns the flattened index of the cell at the specified corner.
            """
            if corner == Corner.TOP_LEFT:
                return self.get_index(0, 0)
            elif corner == Corner.TOP_RIGHT:
                return self.get_index(0, self.cols - 1)
            elif corner == Corner.BOTTOM_RIGHT:
                return self.get_index(self.rows - 1, self.cols - 1)
            elif corner == Corner.BOTTOM_LEFT:
                return self.get_index(self.rows - 1, 0)
            else:
                raise ValueError("Invalid corner")

        def get_index_side(self, side: Side, index: int):
            """
            Returns the flattened index of the cell at the specified side and index.
            """
            if side == Side.TOP:
                return self.get_index(0, index)
            elif side == Side.RIGHT:
                return self.get_index(index, self.cols - 1)
            elif side == Side.BOTTOM:
                return self.get_index(self.rows - 1, index)
            elif side == Side.LEFT:
                return self.get_index(index, 0)
            else:
                raise ValueError("Invalid side")

        def get_row_col(self, index):
            """
            Returns the (row, col) coordinates of the cell at the given flattened index.
            """
            if index < self.cols:
                return int64(0), int64(index)
            elif index < self.cols + self.rows - 1:
                return int64(index - self.cols + 1), int64(self.cols - 1)
            elif index < 2 * self.cols + self.rows - 2:
                return int64(self.rows - 1), int64(
                    2 * self.cols + self.rows - index - 3
                )
            else:
                return int64(2 * self.cols + 2 * self.rows - index - 4), int64(0)

        def size(self):
            """
            Returns the total number of cells in the perimeter.
            """
            return self.data.size

        def get_side(self, side: Side):
            """
            Returns the cells on the specified side of the perimeter.
            """
            if side == Side.TOP:
                # The enitire top edge (from left to right) includes the corners
                return self.data[: self.cols]
            elif side == Side.RIGHT:
                # The enitire right edge (from top to bottom) includes the corners
                return self.data[self.cols - 1 : self.cols + self.rows - 1]
            elif side == Side.BOTTOM:
                # The enitire bottom edge (from left to right) includes the corners
                return self.data[
                    self.cols + self.rows - 2 : 2 * self.cols + self.rows - 2
                ][::-1]  # reversed from internal representation
            elif side == Side.LEFT:
                # The entire left edge (from bottom to top) includes the corners
                indices = np.empty(self.rows, dtype=np.int32)
                indices[0] = 0
                indices[1:] = np.arange(
                    self.data.size - 1, self.data.size - self.rows, -1
                )
                return self.data[indices]
            else:
                raise ValueError("Invalid side")

        def get_corner(self, corner: Corner):
            """
            Returns the cell at the specified corner of the perimeter.
            """
            return self.data[self.get_index_corner(corner)]

    return Perimeter


Float32Perimeter = create_perimeter_class(float32)
Int64Perimeter = create_perimeter_class(int64)
UInt8Perimeter = create_perimeter_class(uint8)


Float32PerimeterList = ListType(Float32Perimeter.class_type.instance_type)
Int64PerimeterList = ListType(Int64Perimeter.class_type.instance_type)
UInt8PerimeterList = ListType(UInt8Perimeter.class_type.instance_type)
