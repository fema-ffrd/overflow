import numpy as np
from numba.experimental import jitclass
from numba.typed import List  # type: ignore[attr-defined]

from overflow._util.constants import (
    FLOW_EXTERNAL,
    NEIGHBOR_OFFSETS,
)
from overflow._util.perimeter import (
    Int64Perimeter,
    Int64PerimeterList,
    UInt8Perimeter,
    UInt8PerimeterList,
)
from overflow._util.queue import Int64PairQueue as Queue


@jitclass
class GlobalState:
    """
    Represents the global state of the flow accumulation calculation.

    This class manages the global state for tiled flow accumulation processing,
    including perimeter data for flow accumulation, flow direction, and link information.

    Attributes:
        num_rows (int): Number of rows in the tile grid.
        num_cols (int): Number of columns in the tile grid.
        chunk_size (int): Size of each tile (chunk) in pixels.
        no_data (float): No data value for the raster.
        flow_acc_perimeters (Int64PerimeterList): List of flow accumulation perimeters for each tile.
        flow_dir_perimeters (UInt8PerimeterList): List of flow direction perimeters for each tile.
        links_row_perimeter (Int64PerimeterList): List of row indices for linked cells.
        links_col_perimeter (Int64PerimeterList): List of column indices for linked cells.
    """

    num_rows: int
    num_cols: int
    chunk_size: int
    no_data: float
    flow_acc_perimeters: Int64PerimeterList
    flow_dir_perimeters: UInt8PerimeterList
    links_row_perimeter: Int64PerimeterList
    links_col_perimeter: Int64PerimeterList

    def __init__(self, num_rows: int, num_cols: int, chunk_size: int, no_data: float):
        """
        Initialize the GlobalState object.

        Args:
            num_rows (int): Number of rows in the tile grid.
            num_cols (int): Number of columns in the tile grid.
            chunk_size (int): Size of each tile (chunk) in pixels.
            no_data (float): No data value for the raster.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.chunk_size = chunk_size
        self.no_data = float(no_data)
        tile_count = num_rows * num_cols

        # Initialize empty perimeter lists for each tile
        self.flow_acc_perimeters = List(
            [Int64Perimeter(np.empty(0, dtype=np.int64), 0, 0, 0)] * tile_count
        )
        self.flow_dir_perimeters = List(
            [UInt8Perimeter(np.empty(0, dtype=np.ubyte), 0, 0, 0)] * tile_count
        )
        self.links_row_perimeter = List(
            [Int64Perimeter(np.empty(0, dtype=np.int64), 0, 0, 0)] * tile_count
        )
        self.links_col_perimeter = List(
            [Int64Perimeter(np.empty(0, dtype=np.int64), 0, 0, 0)] * tile_count
        )

    def update_perimeters(
        self,
        tile_row: int,
        tile_col: int,
        flow_acc_perimeter: np.ndarray,
        flow_dir_perimeter: np.ndarray,
        links_row_perimeter: np.ndarray,
        links_col_perimeter: np.ndarray,
    ):
        """
        Update the perimeter data for a specific tile.

        This method is called after processing each tile to store its perimeter information
        in the global state. This information is crucial for connecting adjacent tiles
        and calculating the global flow accumulation.

        Args:
            tile_row (int): Row index of the tile in the grid.
            tile_col (int): Column index of the tile in the grid.
            flow_acc_perimeter (np.ndarray): Flow accumulation perimeter data.
            flow_dir_perimeter (np.ndarray): Flow direction perimeter data.
            links_row_perimeter (np.ndarray): Row indices for linked cells.
            links_col_perimeter (np.ndarray): Column indices for linked cells.
        """
        tile_index = self.get_tile_index(tile_row, tile_col)
        self.flow_acc_perimeters[tile_index] = Int64Perimeter(
            flow_acc_perimeter, self.chunk_size, self.chunk_size, tile_index
        )
        self.flow_dir_perimeters[tile_index] = UInt8Perimeter(
            flow_dir_perimeter, self.chunk_size, self.chunk_size, tile_index
        )
        self.links_row_perimeter[tile_index] = Int64Perimeter(
            links_row_perimeter, self.chunk_size, self.chunk_size, tile_index
        )
        self.links_col_perimeter[tile_index] = Int64Perimeter(
            links_col_perimeter, self.chunk_size, self.chunk_size, tile_index
        )

    def get_tile_index(self, tile_row: int, tile_col: int) -> int:
        """
        Calculate the tile index from its row and column in the grid.

        This method converts 2D tile coordinates to a 1D index for easier storage and retrieval.

        Args:
            tile_row (int): Row index of the tile.
            tile_col (int): Column index of the tile.

        Returns:
            int: The calculated tile index.
        """
        return tile_row * self.num_cols + tile_col

    def get_next_cell(self, row: int, col: int, flow_direction: int) -> tuple[int, int]:
        """
        Get the next cell coordinates based on the flow direction.

        This method determines the neighboring cell that water would flow into
        given the current cell's flow direction.

        Args:
            row (int): Current cell row.
            col (int): Current cell column.
            flow_direction (int): Flow direction value.

        Returns:
            tuple[int, int]: Next cell row and column.
        """
        # Safety check: prevent out-of-bounds access on NEIGHBOR_OFFSETS (length 8)
        # FLOW_DIRECTION_UNDEFINED = 8, FLOW_DIRECTION_NODATA = 9
        if flow_direction >= 8:
            return row, col
        d_row, d_col = NEIGHBOR_OFFSETS[flow_direction]
        return row + d_row, col + d_col

    def get_global_cell_index(
        self, tile_row: int, tile_col: int, local_row: int, local_col: int
    ) -> int:
        """
        Calculate the global cell index from tile and local coordinates.

        This method converts tile-based coordinates to a global 1D index,
        which is used to uniquely identify each cell across all tiles.

        Args:
            tile_row (int): Tile row in the grid.
            tile_col (int): Tile column in the grid.
            local_row (int): Local row within the tile.
            local_col (int): Local column within the tile.

        Returns:
            int: Global cell index, or -1 if out of bounds.
        """
        # Check if the tile coordinates are within bounds
        if (
            tile_row < 0
            or tile_row >= self.num_rows
            or tile_col < 0
            or tile_col >= self.num_cols
        ):
            return -1

        # Check if the local coordinates are within the chunk size
        if (
            local_row < 0
            or local_row >= self.chunk_size
            or local_col < 0
            or local_col >= self.chunk_size
        ):
            return -1

        # Calculate global row and column
        global_row = tile_row * self.chunk_size + local_row
        global_col = tile_col * self.chunk_size + local_col

        # Convert 2D global coordinates to 1D index
        return global_row * (self.num_cols * self.chunk_size) + global_col

    def calculate_global_accumulation(self):
        """
        Calculate the global accumulation by constructing and solving the global graph.

        This method orchestrates the process of building a global flow graph from tile data
        and then solving it to determine the final flow accumulation values.

        Returns:
            tuple: Global accumulation and offset dictionaries.
        """
        global_graph = self.construct_global_graph()
        global_acc, global_offset = self.solve_global_graph(global_graph)
        return global_acc, global_offset

    def construct_global_graph(self):
        """
        Construct the global graph representing flow connections between tiles.

        This method builds a graph where each node represents a cell on the perimeter of a tile,
        and edges represent the flow connections between these cells, either within the same tile
        or across tile boundaries.

        Returns:
            dict: The constructed global graph.
        """
        global_graph = {}

        for tile_index in range(len(self.flow_acc_perimeters)):
            tile_row = tile_index // self.num_cols
            tile_col = tile_index % self.num_cols
            flow_acc_perimeter = self.flow_acc_perimeters[tile_index]
            flow_dir_perimeter = self.flow_dir_perimeters[tile_index]
            links_row_perimeter = self.links_row_perimeter[tile_index]
            links_col_perimeter = self.links_col_perimeter[tile_index]

            for i in range(flow_acc_perimeter.size()):
                row, col = flow_acc_perimeter.get_row_col(i)
                global_index = self.get_global_cell_index(tile_row, tile_col, row, col)

                # Check if the cell flows to an external tile
                if (
                    links_row_perimeter.data[i] == FLOW_EXTERNAL[0]
                    and links_col_perimeter.data[i] == FLOW_EXTERNAL[1]
                ):
                    flow_direction = flow_dir_perimeter.data[i]
                    next_row, next_col = self.get_next_cell(row, col, flow_direction)
                    next_tile_row, next_tile_col = self.get_neighboring_tile(
                        tile_row, tile_col, next_row, next_col
                    )
                    next_local_row, next_local_col = self.adjust_coordinates(
                        next_row, next_col
                    )
                    next_global_index = self.get_global_cell_index(
                        next_tile_row, next_tile_col, next_local_row, next_local_col
                    )
                    # Add to graph if this is not the same cell (not nodata)
                    if next_global_index != global_index:
                        global_graph[global_index] = {
                            "accumulation": flow_acc_perimeter.data[i],
                            "next": next_global_index,
                            "is_external": True,
                        }
                else:
                    # Cell flows within the same tile
                    next_row, next_col = (
                        links_row_perimeter.data[i],
                        links_col_perimeter.data[i],
                    )
                    next_global_index = self.get_global_cell_index(
                        tile_row, tile_col, next_row, next_col
                    )
                    # Add to graph if this is not the same cell (not nodata)
                    if next_global_index != global_index:
                        global_graph[global_index] = {
                            "accumulation": flow_acc_perimeter.data[i],
                            "next": next_global_index,
                            "is_external": False,
                        }

        return global_graph

    def solve_global_graph(self, global_graph):
        """
        Solve the global graph to calculate accumulation and offsets.

        This method implements a topological sort-like algorithm to propagate flow
        accumulation through the graph. It starts from source nodes (cells with no inflow)
        and propagates the accumulation downstream.

        Args:
            global_graph (dict): The global graph representing flow connections.

        Returns:
            tuple: Global accumulation and offset dictionaries.
        """
        global_acc = {}
        global_offset = {}
        inflow_count = {}

        # Initialize dictionaries
        for index in global_graph:
            inflow_count[index] = 0
            global_acc[index] = 0
            global_offset[index] = 0

        # Count inflows and set accumulation for external nodes
        for index, node in global_graph.items():
            if node["next"] in inflow_count:
                inflow_count[node["next"]] += 1
            if node["is_external"]:
                global_acc[index] = node["accumulation"]

        # Initialize queue with source nodes (cells with no inflow)
        queue = Queue([(0, 0)])
        queue.pop()  # Clear the initial dummy value
        for index in global_graph:
            if inflow_count[index] == 0:
                queue.push((index, 0))

        # Propagate flow downstream using the queue
        while queue:
            index, current_offset = queue.pop()
            node = global_graph[index]

            # Update offset for the current node
            current_offset += global_acc[index]

            # Propagate to next node
            if node["next"] in global_graph:
                next_index = node["next"]
                global_offset[next_index] += current_offset
                inflow_count[next_index] -= 1
                if inflow_count[next_index] == 0:
                    # All inflows processed, add to queue
                    queue.push((next_index, global_offset[next_index]))

        return global_acc, global_offset

    def get_neighboring_tile(
        self, tile_row: int, tile_col: int, row: int, col: int
    ) -> tuple[int, int]:
        """
        Get the neighboring tile based on the current tile and cell coordinates.

        This method determines which adjacent tile a cell flows into when it's on the edge of its current tile.

        Args:
            tile_row (int): Current tile row.
            tile_col (int): Current tile column.
            row (int): Cell row within the tile.
            col (int): Cell column within the tile.

        Returns:
            tuple[int, int]: Neighboring tile row and column.
        """
        next_tile_row = tile_row
        next_tile_col = tile_col
        if row < 0:
            next_tile_row -= 1
        elif row >= self.chunk_size:
            next_tile_row += 1
        if col < 0:
            next_tile_col -= 1
        elif col >= self.chunk_size:
            next_tile_col += 1
        return next_tile_row, next_tile_col

    def adjust_coordinates(self, row: int, col: int) -> tuple[int, int]:
        """
        Adjust coordinates to be within the chunk size.

        This method is used when a cell flows into a neighboring tile, to convert its
        coordinates to be relative to the new tile.

        Args:
            row (int): Row coordinate to adjust.
            col (int): Column coordinate to adjust.

        Returns:
            tuple[int, int]: Adjusted row and column coordinates.
        """
        adjusted_row = row % self.chunk_size
        adjusted_col = col % self.chunk_size
        return adjusted_row, adjusted_col
