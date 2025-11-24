import numpy as np
from numba import int64, uint8  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import Dict  # type: ignore[attr-defined]
from numba.types import DictType

from overflow._util.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_WEST,
)
from overflow._util.perimeter import Int64Perimeter, UInt8Perimeter
from overflow._util.raster import Corner, Side


@jitclass
class TileEdgeData:
    """
    A class representing the edge data of a tile. Used to organize
    The perimeter data of a tile for processing.

    Attributes:
    - flow_direction (UInt8Perimeter): The flow direction data for the tile edges.
    - watershed (Int64Perimeter): The watershed data for the tile edges.
    """

    flow_direction: UInt8Perimeter
    watershed: Int64Perimeter

    def __init__(self, flow_direction: UInt8Perimeter, watershed: Int64Perimeter):
        self.flow_direction = flow_direction
        self.watershed = watershed


@jitclass(
    [
        ("flow_directions", uint8[:, :]),
        ("watersheds", int64[:, :]),
        ("graph", DictType(int64, int64)),
    ]
)
class GlobalState:
    """
    A class representing the global state of the flow direction and watershed data.
    It is used to store a global graph of watershed connections between tiles.

    Attributes:
    - flow_directions (np.ndarray): A 2D array storing the flow direction data for each tile.
    - watersheds (np.ndarray): A 2D array storing the watershed data for each tile.
    - tile_rows (int): The number of rows in the tile grid.
    - tile_cols (int): The number of columns in the tile grid.
    - chunk_size (int): The size of each tile chunk.
    - graph (dict): A dictionary representing the graph of watershed connections.

    Methods:
    - __init__(self, tile_rows: int, tile_cols: int, chunk_size: int):
        Initializes the GlobalState object with the given tile grid dimensions and chunk size.
    - _get_tile_index(self, row: int, col: int) -> int:
        Computes the index of a tile in the 1D arrays based on its row and column.
    - _get_directions_perimeter(self, tile_row: int, tile_col: int) -> UInt8Perimeter:
        Retrieves the flow direction perimeter for a specific tile.
    - _get_watersheds_perimeter(self, tile_row: int, tile_col: int) -> Int64Perimeter:
        Retrieves the watershed perimeter for a specific tile.
    - _get_tile_edge_data(self, tile_row: int, tile_col: int) -> TileEdgeData:
        Retrieves the edge data (flow direction and watershed perimeters) for a specific tile.
    - _handle_corner(self, tile_a: TileEdgeData, tile_b: TileEdgeData, corner_a: Corner):
        Handles the flow direction and watershed connections at a corner between two tiles.
    - _handle_edge(self, tile_a: TileEdgeData, tile_b: TileEdgeData, side_a: Side):
        Handles the flow direction and watershed connections at an edge between two tiles.
    - _join_adjacent_tiles(self, tile_a: TileEdgeData, tile_b: TileEdgeData, tile_c: TileEdgeData,
        tile_d: TileEdgeData): Joins the adjacent tiles by handling the flow direction and watershed connections at
        their edges and corners.
    - complete_graph(self):
        Completes the graph of watershed connections by iterating over all tiles and joining adjacent tiles.
    """

    flow_directions: np.ndarray
    watersheds: np.ndarray
    tile_rows: int
    tile_cols: int
    chunk_size: int
    graph: dict

    def __init__(self, tile_rows: int, tile_cols: int, chunk_size: int):
        perimeter_cells_per_tile = 2 * (chunk_size + chunk_size) - 4
        tile_count = tile_rows * tile_cols
        self.flow_directions = np.empty(
            (tile_count, perimeter_cells_per_tile), dtype=np.uint8
        )
        self.watersheds = np.empty(
            (tile_count, perimeter_cells_per_tile), dtype=np.int64
        )
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.chunk_size = chunk_size
        self.graph = Dict.empty(int64, int64)

    def _get_tile_index(self, row: int, col: int) -> int:
        return row * self.tile_cols + col

    def _get_directions_perimeter(self, tile_row: int, tile_col: int) -> UInt8Perimeter:
        tile_index = self._get_tile_index(tile_row, tile_col)
        return UInt8Perimeter(
            self.flow_directions[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def _get_watersheds_perimeter(self, tile_row: int, tile_col: int) -> Int64Perimeter:
        tile_index = self._get_tile_index(tile_row, tile_col)
        return Int64Perimeter(
            self.watersheds[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def _get_tile_edge_data(self, tile_row: int, tile_col: int) -> TileEdgeData:
        return TileEdgeData(
            self._get_directions_perimeter(tile_row, tile_col),
            self._get_watersheds_perimeter(tile_row, tile_col),
        )

    def _handle_corner(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        corner_a: Corner,
    ):
        if corner_a == Corner.TOP_LEFT:
            direction_a = FLOW_DIRECTION_NORTH_WEST
            corner_b = Corner.BOTTOM_RIGHT
        if corner_a == Corner.TOP_RIGHT:
            direction_a = FLOW_DIRECTION_NORTH_EAST
            corner_b = Corner.BOTTOM_LEFT
        if corner_a == Corner.BOTTOM_LEFT:
            direction_a = FLOW_DIRECTION_SOUTH_WEST
            corner_b = Corner.TOP_RIGHT
        if corner_a == Corner.BOTTOM_RIGHT:
            direction_a = FLOW_DIRECTION_SOUTH_EAST
            corner_b = Corner.TOP_LEFT

        if tile_a.flow_direction.get_corner(corner_a) == direction_a:
            self.graph[tile_a.watershed.get_corner(corner_a)] = (
                tile_b.watershed.get_corner(corner_b)
            )

    def _handle_edge(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        side_a: Side,
    ):
        if side_a == Side.RIGHT:
            flow_across = FLOW_DIRECTION_EAST
            flow_up = FLOW_DIRECTION_NORTH_EAST
            flow_down = FLOW_DIRECTION_SOUTH_EAST
            side_b = Side.LEFT
        elif side_a == Side.BOTTOM:
            flow_across = FLOW_DIRECTION_SOUTH
            flow_up = FLOW_DIRECTION_SOUTH_WEST
            flow_down = FLOW_DIRECTION_SOUTH_EAST
            side_b = Side.TOP
        elif side_a == Side.LEFT:
            flow_across = FLOW_DIRECTION_WEST
            flow_up = FLOW_DIRECTION_NORTH_WEST
            flow_down = FLOW_DIRECTION_SOUTH_WEST
            side_b = Side.RIGHT
        elif side_a == Side.TOP:
            flow_across = FLOW_DIRECTION_NORTH
            flow_up = FLOW_DIRECTION_NORTH_WEST
            flow_down = FLOW_DIRECTION_NORTH_EAST
            side_b = Side.BOTTOM
        # check if tile a flows into tile b
        tile_a_flow_directions_side = tile_a.flow_direction.get_side(side_a)
        for i, direction in enumerate(tile_a_flow_directions_side):
            if direction == flow_across:
                self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                    tile_b.watershed.get_side(side_b)[i]
                )
            if i < len(tile_a_flow_directions_side) - 1:
                if direction == flow_down:
                    self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                        tile_b.watershed.get_side(side_b)[i + 1]
                    )
            if i > 0 and direction == flow_up:
                self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                    tile_b.watershed.get_side(side_b)[i - 1]
                )

    def _join_adjacent_tiles(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        tile_c: TileEdgeData,
        tile_d: TileEdgeData,
    ):
        """
        Connect the edges and corners of adjacent tiles.
        Between the four adjacent tiles, the connections are as follows:
        + - - + - - +
        |  A  |  B  |
        + - - * - - +
        |  C  |  D  |
        + - - + - - +
        A and B are connected at their adjacent edge.
        B and D are connected at their adjacent edge.
        D and C are connected at their adjacent edge.
        C and A are connected at their adjacent edge.
        A and D are connected at their corners.
        C and B are connected at their corners.
        """
        self._handle_corner(tile_a, tile_d, Corner.BOTTOM_RIGHT)
        self._handle_corner(tile_d, tile_a, Corner.TOP_LEFT)
        self._handle_corner(tile_b, tile_c, Corner.BOTTOM_LEFT)
        self._handle_corner(tile_c, tile_b, Corner.TOP_RIGHT)
        self._handle_edge(tile_a, tile_b, Side.RIGHT)
        self._handle_edge(tile_b, tile_a, Side.LEFT)
        self._handle_edge(tile_c, tile_d, Side.RIGHT)
        self._handle_edge(tile_d, tile_c, Side.LEFT)
        self._handle_edge(tile_a, tile_c, Side.BOTTOM)
        self._handle_edge(tile_c, tile_a, Side.TOP)
        self._handle_edge(tile_b, tile_d, Side.BOTTOM)
        self._handle_edge(tile_d, tile_b, Side.TOP)

    def complete_graph(self):
        """
        Complete the global graph by joining adjacent tiles.
        """
        for tile_row in range(self.tile_rows - 1):
            for tile_col in range(self.tile_cols - 1):
                tile_a = self._get_tile_edge_data(tile_row, tile_col)
                tile_b = self._get_tile_edge_data(tile_row, tile_col + 1)
                tile_c = self._get_tile_edge_data(tile_row + 1, tile_col)
                tile_d = self._get_tile_edge_data(tile_row + 1, tile_col + 1)
                self._join_adjacent_tiles(tile_a, tile_b, tile_c, tile_d)
