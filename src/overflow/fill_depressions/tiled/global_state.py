from heapq import heappop, heappush
import numpy as np
from numba import njit
from numba.types import int64, float32
from numba.typed import Dict  # pylint: disable=no-name-in-module
from numba.experimental import jitclass
from overflow.util.raster import Side, Corner
from overflow.util.perimeter import (
    Int64Perimeter,
    Float32Perimeter,
)
from overflow.fill_depressions.core.watershed_graph import WatershedGraph


@njit
def handle_edge(
    dem_a: np.ndarray,
    labels_a: np.ndarray,
    dem_b: np.ndarray,
    labels_b: np.ndarray,
    graph: WatershedGraph,
    no_data: float32,
) -> None:
    """
    Combine two tiles by joining their edges.
    Algorithm 2 from Parallel Priority-Flood (R. Barnes)
    https://arxiv.org/pdf/1606.06204.pdf

    Args:
        dem_a (np.ndarray): Vector of cell elevations from tile A adjacent to tile B.
        labels_a (np.ndarray): Vector of cell labels from tile A adjacent to tile B.
        dem_b (np.ndarray): Vector of cell elevations from tile B adjacent to tile A.
        labels_b (np.ndarray): Vector of cell labels from tile B adjacent to tile A.
        graph (GlobalGraph): Master graph containing the
            partially-joined graphs of all tiles.
        no_data (float32): The value representing no data in the DEM.

    Returns:
        None. The function modifies the graph in place.
    """
    # Iterate over all indices in the length of DEM_A
    for i, elev_a in enumerate(dem_a):
        elev_a = (
            elev_a if elev_a != no_data and not np.isnan(elev_a) else float32(-np.inf)
        )
        # Iterate over all neighboring indices
        for ni in [i - 1, i, i + 1]:
            elev_b = dem_b[ni] if 0 <= ni < dem_b.shape[0] else float32(-np.inf)
            # Skip if the neighboring index is out of bounds
            if ni < 0 or ni == dem_a.shape[0]:
                continue
            # Skip if the labels at the current index and the neighboring index are the same
            label_a = labels_a[i]
            label_b = labels_b[ni]
            if label_a == label_b:
                continue
            # Calculate the maximum elevation between the current cell and the neighboring cell
            e = max(elev_a, elev_b)
            # update the graph
            if (label_a, label_b) not in graph or e < graph[(label_a, label_b)]:
                graph[(label_a, label_b)] = e


@njit
def handle_corner(
    elev_a: float32,
    label_a: int64,
    elev_b: float32,
    label_b: int64,
    graph: WatershedGraph,
    no_data: float32,
) -> None:
    """
    Combine two tiles by joining their corners.
    Algorithm 2 analog from Parallel Priority-Flood (R. Barnes)
    https://arxiv.org/pdf/1606.06204.pdf

    Args:
        elev_a (float32): Elevation from tile A corner adjacent to tile B corner.
        label_a (int64): Label from tile A corner adjacent to tile B corner.
        elev_b (float32): Elevation from tile B corner adjacent to tile A corner.
        label_b (int64): Label from tile B corner adjacent to tile A corner.
        graph (GlobalGraph): Master graph containing the
            partially-joined graphs of all tiles.

    Returns:
        None. The function modifies the graph in place.
    """
    # Handle no_data values
    elev_a = elev_a if elev_a != no_data and not np.isnan(elev_a) else float32(-np.inf)
    elev_b = elev_b if elev_b != no_data and not np.isnan(elev_b) else float32(-np.inf)

    # Skip if the labels at the corner of tile A and tile B are the same
    if label_a == label_b:
        return

    # Calculate the maximum elevation between the corner cell of tile A and tile B
    e = max(elev_a, elev_b)

    # Update the graph
    if (label_a, label_b) not in graph or e < graph[(label_a, label_b)]:
        graph[(label_a, label_b)] = e


@jitclass
class PerimeterData:
    """
    Container for label and elevation perimeters of a tile.

    Attributes:
    - labels (Int64Perimeter): Perimeter of labels for the tile.
    - elevations (Float32Perimeter): Perimeter of elevations for the tile.
    """

    labels: Int64Perimeter
    elevations: Float32Perimeter

    def __init__(
        self, label_perimeters: Int64Perimeter, elevation_perimeters: Float32Perimeter
    ):
        self.labels = label_perimeters
        self.elevations = elevation_perimeters


@jitclass(
    [
        ("label_perimeters", int64[:, :]),
        ("elevation_perimeters", float32[:, :]),
    ]
)
class GlobalState:
    """
    A class representing the global state of the flow direction and watershed data.
    It is used to store a global graph of watershed connections between tiles.

    Attributes:
    - graph (WatershedGraph): The global graph of watershed connections.
    - label_perimeters (np.ndarray): A 2D array storing the label perimeters for each tile.
    - elevation_perimeters (np.ndarray): A 2D array storing the elevation perimeters for each tile.
    - num_rows (int64): The number of rows in the tile grid.
    - num_cols (int64): The number of columns in the tile grid.
    - chunk_size (int64): The size of each tile chunk.
    - no_data (float32): The value representing no data in the DEM.

    Methods:
    - __init__(self, num_rows: int64, num_cols: int64, chunk_size: int64, no_data: float32):
        Initializes the GlobalState object with the given tile grid dimensions, chunk size, and no data value.
    - _row_col_to_tile_index(self, row: int64, col: int64) -> int64:
        Computes the index of a tile in the 1D arrays based on its row and column.
    - _get_tile_perimeter(self, tile_index: int64) -> PerimeterData:
        Retrieves the perimeter data for a specific tile.
    - connect_tile_edges_and_corners(self):
        Connects the edges and corners of the tiles in the global state.
    - _combine_edge(
        self,
        perimeter_data_a: PerimeterData,
        side_a: Side,
        perimeter_data_b: PerimeterData,
        side_b: Side,
    ):
        Combines two tiles by joining their edges.
    - _combine_corner(
        self,
        perimeter_data_a: PerimeterData,
        corner_a: Corner,
        perimeter_data_b: PerimeterData,
        corner_b: Corner,
    ):
        Combines two tiles by joining their corners.
    - solve_graph(self):
        Solves the global graph of watershed connections using the priority flood algorithm.
    """

    graph: WatershedGraph
    label_perimeters: np.ndarray
    elevation_perimeters: np.ndarray
    num_rows: int64
    num_cols: int64
    chunk_size: int64
    no_data: float32

    def __init__(
        self, num_rows: int64, num_cols: int64, chunk_size: int64, no_data: float32
    ):
        self.graph = WatershedGraph()
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.chunk_size = chunk_size
        self.no_data = float32(no_data)
        tile_count = num_rows * num_cols
        perimeter_cell_count = 4 * chunk_size - 4

        self.label_perimeters = np.empty(
            (tile_count, perimeter_cell_count), dtype=np.int64
        )
        self.elevation_perimeters = np.empty(
            (tile_count, perimeter_cell_count), dtype=np.float32
        )

    def _row_col_to_tile_index(self, row: int64, col: int64) -> int64:
        return row * self.num_cols + col

    def _get_tile_perimeter(self, tile_index: int64) -> PerimeterData:
        labels = Int64Perimeter(
            self.label_perimeters[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )
        elevations = Float32Perimeter(
            self.elevation_perimeters[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )
        return PerimeterData(labels, elevations)

    def connect_tile_edges_and_corners(self):
        """Combine the spillover graphs of all tiles into a single global graph."""
        for row_index in range(self.num_rows - 1):
            for col_index in range(self.num_cols - 1):
                # + - - + - - +
                # |  A  |  B  |
                # + - - * - - +
                # |  C  |  D  |
                # + - - + - - +
                tile_index_a = self._row_col_to_tile_index(row_index, col_index)
                tile_index_b = self._row_col_to_tile_index(row_index, col_index + 1)
                tile_index_c = self._row_col_to_tile_index(row_index + 1, col_index)
                tile_index_d = self._row_col_to_tile_index(row_index + 1, col_index + 1)
                perimeter_data_a = self._get_tile_perimeter(tile_index_a)
                perimeter_data_b = self._get_tile_perimeter(tile_index_b)
                perimeter_data_c = self._get_tile_perimeter(tile_index_c)
                perimeter_data_d = self._get_tile_perimeter(tile_index_d)

                # Connect A-B, A-C, D-B, D-C edges
                self._combine_edge(
                    perimeter_data_a, Side.RIGHT, perimeter_data_b, Side.LEFT
                )
                self._combine_edge(
                    perimeter_data_a, Side.BOTTOM, perimeter_data_c, Side.TOP
                )
                self._combine_edge(
                    perimeter_data_d, Side.TOP, perimeter_data_b, Side.BOTTOM
                )
                self._combine_edge(
                    perimeter_data_d, Side.LEFT, perimeter_data_c, Side.RIGHT
                )
                # Connect A-D and B-C corners
                self._combine_corner(
                    perimeter_data_a,
                    Corner.BOTTOM_RIGHT,
                    perimeter_data_d,
                    Corner.TOP_LEFT,
                )
                self._combine_corner(
                    perimeter_data_b,
                    Corner.BOTTOM_LEFT,
                    perimeter_data_c,
                    Corner.TOP_RIGHT,
                )

    def _combine_edge(
        self,
        perimeter_data_a: PerimeterData,
        side_a: Side,
        perimeter_data_b: PerimeterData,
        side_b: Side,
    ):
        """Combine two tiles by joining their edges."""
        dem_a = perimeter_data_a.elevations.get_side(side_a)
        labels_a = perimeter_data_a.labels.get_side(side_a)
        dem_b = perimeter_data_b.elevations.get_side(side_b)
        labels_b = perimeter_data_b.labels.get_side(side_b)
        handle_edge(dem_a, labels_a, dem_b, labels_b, self.graph, self.no_data)

    def _combine_corner(
        self,
        perimeter_data_a: PerimeterData,
        corner_a: Corner,
        perimeter_data_b: PerimeterData,
        corner_b: Corner,
    ):
        """Combine two tiles by joining their corners."""
        dem_a = perimeter_data_a.elevations.get_corner(corner_a)
        labels_a = perimeter_data_a.labels.get_corner(corner_a)
        dem_b = perimeter_data_b.elevations.get_corner(corner_b)
        labels_b = perimeter_data_b.labels.get_corner(corner_b)
        handle_corner(dem_a, labels_a, dem_b, labels_b, self.graph, self.no_data)

    def solve_graph(self):
        """
        Modified priority flood algorithm for graph

        modifies mastergraph and elevation_graph in place

        Algorithm 2 from priority-flood (r. barnes, 2015)
        https://arxiv.org/pdf/1511.04463.pdf
        """
        open_heap = [(float32(-np.inf), int64(1))]
        closed_set = set()
        label_min_elevations = Dict.empty(key_type=int64, value_type=float32)
        label_min_elevations[1] = float32(-np.inf)
        while open_heap:
            elev, node = heappop(open_heap)
            if node in closed_set:
                continue
            closed_set.add(node)
            label_min_elevations[node] = elev
            for neighbor, neighbor_elev in self.graph.neighbors(node).items():
                if neighbor not in closed_set:
                    max_height = max(elev, neighbor_elev)
                    new_node = (max_height, neighbor)
                    heappush(open_heap, new_node)

        return label_min_elevations
