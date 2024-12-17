import heapq
from numba.experimental import jitclass
from numba.typed import List  # pylint: disable=no-name-in-module
from numba.types import Array
from numba import int64
import numpy as np
from overflow.util.numba_types import Int64PairListList, Int64PairList, Int64Pair
from overflow.util.raster import Corner, Side
from overflow.util.perimeter import Float32Perimeter


@jitclass
class LocalGraph:
    """
    Represents a local graph for a tile in the DEM.

    This class constructs a local graph for a tile by connecting neighboring cells in the same flat
    and connecting each cell to the high/low terrain nodes. It also reduces unnecessary edges by
    skipping edges where shorter paths exist between the same vertices.

    Attributes:
        flat_edges (Int64PairListList): A list of lists containing the edges connecting cells in the same flat.
        high_edges (Int64PairList): A list of edges connecting cells to the high terrain node.
        low_edges (Int64PairList): A list of edges connecting cells to the low terrain node.
        tile_index (int): The index of the tile in the global DEM.
    """

    flat_edges: Int64PairListList
    high_edges: Int64PairList
    low_edges: Int64PairList
    tile_index: int

    def __init__(
        self,
        labels_perimeter: np.ndarray,
        to_higher_perimeter: np.ndarray,
        to_lower_perimeter: np.ndarray,
        min_dists: np.ndarray,
        tile_index: int,
    ):
        self.tile_index = tile_index
        cell_count = labels_perimeter.size
        self.flat_edges = self._to_shortest_adjacency_list(min_dists)
        index_offset = self.tile_index * cell_count

        # connect each cell to the high/low terrain node
        self.high_edges = List.empty_list(Int64Pair)
        self.low_edges = List.empty_list(Int64Pair)
        for i, distance in enumerate(to_higher_perimeter):
            if distance == 0:
                continue
            self.high_edges.append((i + index_offset, distance))

        for i, distance in enumerate(to_lower_perimeter):
            if distance == 0:
                continue
            self.low_edges.append((i + index_offset, distance))

    def _to_shortest_adjacency_list(self, dist: np.ndarray) -> List:
        """
        Constructs an optimized adjacency list representation of the graph connecting all
        flat cells along the perimeter of the tile. The adjacency list is constructed using
        the min_dists array, which is an adjacency matrix containing the shortest distance
        from any perimeter cell to any other perimeter cell.

        The optimization process involves removing redundant edges that do not contribute to
        the shortest paths between nodes. For each pair of cells (i, j), the method checks
        if there exists a shorter path through another cell k. If such a path exists, the
        direct edge between i and j is considered redundant and is not added to the adjacency
        list. This optimization ensures that the adjacency list contains the least amount of
        edges necessary to represent the shortest paths between cells, reducing the complexity
        of the graph.

        Args:
            dist (np.ndarray): The min_dists array containing the shortest distances
                between perimeter cells.

        Returns:
            List: The optimized adjacency list representation of the graph, where each element
                is a list of tuples (j, distance) representing an edge from cell i to cell j
                with the given distance.
        """
        cell_count = dist.shape[0]
        adjacency_list = List.empty_list(Int64PairList)
        for _ in range(cell_count):
            adjacency_list.append(List.empty_list(Int64Pair))

        for i in range(cell_count):
            for j in range(i + 1, cell_count):
                if dist[i][j] != 0:
                    is_direct_edge = True
                    for k in range(cell_count):
                        if k != i and k != j and dist[i][k] + dist[k][j] == dist[i][j]:
                            is_direct_edge = False
                            break
                    if is_direct_edge:
                        # add two edges since we don't know from which direction we will traverse
                        adjacency_list[i].append(
                            (j + self.tile_index * cell_count, dist[i][j])
                        )
                        adjacency_list[j].append(
                            (i + self.tile_index * cell_count, dist[i][j])
                        )

        return adjacency_list


@jitclass
class GlobalGraph:
    """
    Represents the global graph of the DEM.

    This class constructs the global graph by connecting adjacent tiles and solving the shortest
    paths from each cell to the high and low terrain nodes using Dijkstra's algorithm.

    Attributes:
        flat_edges (Int64PairListList): A list of lists containing the edges connecting cells in the same flat.
        high_edges (Int64PairList): A list of edges connecting cells to the high terrain node.
        low_edges (Int64PairList): A list of edges connecting cells to the low terrain node.
        solved_high_edges (Array(int64, 1, "C")): An array containing the solved distances from each cell to the high terrain node.
        solved_low_edges (Array(int64, 1, "C")): An array containing the solved distances from each cell to the low terrain node.
        tile_rows (int): The number of rows of tiles in the DEM.
        tile_cols (int): The number of columns of tiles in the DEM.
        perimeter_cells_per_tile (int): The number of perimeter cells per tile.
    """

    flat_edges: Int64PairListList
    high_edges: Int64PairList
    low_edges: Int64PairList
    solved_high_edges: Array(int64, 1, "C")
    solved_low_edges: Array(int64, 1, "C")
    tile_rows: int
    tile_cols: int
    perimeter_cells_per_tile: int

    def __init__(self, tile_rows: int, tile_cols: int, perimeter_cells_per_tile: int):
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.perimeter_cells_per_tile = perimeter_cells_per_tile
        self.flat_edges = List.empty_list(Int64PairList)
        size = tile_rows * tile_cols * perimeter_cells_per_tile
        for _ in range(size):
            self.flat_edges.append(List.empty_list(Int64Pair))
        self.high_edges = List.empty_list(Int64Pair)
        self.low_edges = List.empty_list(Int64Pair)
        self.solved_high_edges = np.zeros(size, dtype=np.int64)
        self.solved_low_edges = np.zeros(size, dtype=np.int64)

    def add(self, local_graph: LocalGraph):
        """
        Add a LocalGraph to the GlobalGraph.

        Args:
            local_graph (LocalGraph): The LocalGraph object to be added.
        """
        for local_index, edge in enumerate(local_graph.flat_edges):
            global_index = local_index + local_graph.tile_index * len(
                local_graph.flat_edges
            )
            self.flat_edges[global_index] = edge
        self.high_edges.extend(local_graph.high_edges)
        self.low_edges.extend(local_graph.low_edges)

    def _join_neighbors(self, i: int, j: int):
        self.flat_edges[i].append((j, 1))
        self.flat_edges[j].append((i, 1))

    def _handle_edge(
        self,
        elevations_a: Float32Perimeter,
        elevations_b: Float32Perimeter,
        side_a: Side,
        side_b: Side,
    ):
        """
        Handle the edge between two adjacent tiles.

        Args:
            elevations_a (Float32Perimeter): The elevation perimeter of the first tile.
            elevations_b (Float32Perimeter): The elevation perimeter of the second tile.
            side_a (Side): The side of the first tile that is adjacent to the second tile.
            side_b (Side): The side of the second tile that is adjacent to the first tile.
        """
        elevations_a_side = elevations_a.get_side(side_a)
        elevations_b_side = elevations_b.get_side(side_b)
        for i, elev_a in enumerate(elevations_a_side):
            elev_b = elevations_b_side[i]
            global_index_a = (
                elevations_a.get_index_side(side_a, i) + elevations_a.index_offset
            )
            # if the neighboring cell directly adjacent is part of the same flat, connect this cell to it
            if elev_a == elev_b:
                global_index_b = (
                    elevations_b.get_index_side(side_b, i) + elevations_b.index_offset
                )
                self._join_neighbors(global_index_a, global_index_b)
            # if the neighboring cell diagonally is part of the same flat, connect this cell to it
            if i != 0 and elev_a == elevations_b_side[i - 1]:
                global_index_b = (
                    elevations_b.get_index_side(side_b, i - 1)
                    + elevations_b.index_offset
                )
                self._join_neighbors(global_index_a, global_index_b)
            if i != elevations_a_side.size - 1 and elev_a == elevations_b_side[i + 1]:
                global_index_b = (
                    elevations_b.get_index_side(side_b, i + 1)
                    + elevations_b.index_offset
                )
                self._join_neighbors(global_index_a, global_index_b)

    def _handle_corner(
        self,
        elevations_a: Float32Perimeter,
        elevations_b: Float32Perimeter,
        corner_a: Corner,
        corner_b: Corner,
    ):
        """
        Handle the corner between two adjacent tiles.

        Args:
            elevations_a (Float32Perimeter): The elevation perimeter of the first tile.
            elevations_b (Float32Perimeter): The elevation perimeter of the second tile.
            corner_a (Corner): The corner of the first tile that is adjacent to the second tile.
            corner_b (Corner): The corner of the second tile that is adjacent to the first tile.
        """
        elev_a = elevations_a.get_corner(corner_a)
        elev_b = elevations_b.get_corner(corner_b)
        if elev_a == elev_b:
            global_index_a = (
                elevations_a.get_index_corner(corner_a) + elevations_a.index_offset
            )
            global_index_b = (
                elevations_b.get_index_corner(corner_b) + elevations_b.index_offset
            )
            self._join_neighbors(global_index_a, global_index_b)

    def join_adjacent_tiles(
        self,
        elevations_a: Float32Perimeter,
        elevations_b: Float32Perimeter,
        elevations_c: Float32Perimeter,
        elevations_d: Float32Perimeter,
    ):
        """
        Join adjacent tiles in the global graph.

        Args:
            elevations_a (Float32Perimeter): The elevation perimeter of tile A.
            elevations_b (Float32Perimeter): The elevation perimeter of tile B.
            elevations_c (Float32Perimeter): The elevation perimeter of tile C.
            elevations_d (Float32Perimeter): The elevation perimeter of tile D.

        The adjacent tiles are represented as follows:
        + - - + - - +
        |  A  |  B  |
        + - - * - - +
        |  C  |  D  |
        + - - + - - +
        """
        # connect edge A-B
        self._handle_edge(elevations_a, elevations_b, Side.RIGHT, Side.LEFT)
        # connect edge B-D
        self._handle_edge(elevations_b, elevations_d, Side.BOTTOM, Side.TOP)
        # connect edge D-C
        self._handle_edge(elevations_d, elevations_c, Side.LEFT, Side.RIGHT)
        # connect edge C-A
        self._handle_edge(elevations_c, elevations_a, Side.TOP, Side.BOTTOM)
        # connect corner A-D
        self._handle_corner(
            elevations_a, elevations_d, Corner.BOTTOM_RIGHT, Corner.TOP_LEFT
        )
        # connect corner B-C
        self._handle_corner(
            elevations_b, elevations_c, Corner.BOTTOM_LEFT, Corner.TOP_RIGHT
        )

    def _init_heap(self, edges: Int64PairList, solved_edges: np.ndarray) -> list:
        """
        Initialize the priority queue for Dijkstra's algorithm.

        Args:
            edges (Int64PairList): The edges to be added to the priority queue.
            solved_edges (Array(int64, 1, "C")): The array to store the solved distances.

        Returns:
            List(Int64Pair): The initialized priority queue.
        """
        pq = List.empty_list(Int64Pair)
        for neighbor, weight in edges:
            # ignore low edge cells for distance to high edge calculation
            if self.solved_low_edges[neighbor] == 1:
                continue
            solved_edges[neighbor] = weight
            heapq.heappush(pq, (weight, neighbor))
        return pq

    def _djikstra_dist_to_edge(self, edge_type: str):
        """
        Calculate the shortest distances from each cell to the specified edge type using Dijkstra's algorithm.

        Args:
            edge_type (str): The type of edge to calculate distances to. Can be "low" or "high".
        """
        solved_edges = (
            self.solved_low_edges if edge_type == "low" else self.solved_high_edges
        )
        pq = self._init_heap(
            self.low_edges if edge_type == "low" else self.high_edges, solved_edges
        )
        while pq:
            dist, node = heapq.heappop(pq)
            if 0 < solved_edges[node] < dist:
                continue
            for neighbor, weight in self.flat_edges[node]:
                # ignore low edge cells for distance to high edge calculation
                if self.solved_low_edges[neighbor] == 1:
                    continue
                if (
                    solved_edges[neighbor] == 0
                    or dist + weight < solved_edges[neighbor]
                ):
                    solved_edges[neighbor] = dist + weight
                    heapq.heappush(pq, (dist + weight, neighbor))
        if edge_type == "low":
            self.solved_low_edges = solved_edges
        else:
            self.solved_high_edges = solved_edges

    def solve_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the global graph by calculating the shortest distances from each cell to the high and low terrain nodes.

        Returns:
            tuple: A tuple containing two 2D arrays of the solved high and low edge distances for each tile.
        """
        self._djikstra_dist_to_edge("low")
        self._djikstra_dist_to_edge("high")
        tile_count = self.tile_rows * self.tile_cols
        # return 2D arrays of the solved high and low edge distances
        return self.solved_low_edges.reshape(
            (tile_count, self.perimeter_cells_per_tile)
        ), self.solved_high_edges.reshape((tile_count, self.perimeter_cells_per_tile))
