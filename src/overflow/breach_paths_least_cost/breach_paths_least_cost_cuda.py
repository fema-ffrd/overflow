## This module, breach_paths_least_cost_cuda.py, is an experimental implementation of the least
## cost path algorithm for breaching pits in a Digital Elevation Model (DEM). It uses CUDA for
## parallel processing on a GPU. The module contains one main function: breach_paths_least_cost_cuda
## in addition to several supporting functions that are used by the main function.
##
## Memory Requirements
## The memory requirements for this module are primarily determined by three parameters:
##      - chunk_size
##      - search_radius
##      - max_pits
##
## chunk_size: This parameter determines the size of the chunks into which the input raster is
## divided for processing. Larger chunk sizes will require more memory but can potentially improve
## performance by reducing the overhead of launching CUDA kernels and overhead from copying data
## to / from the GPU. The memory required for storing the chunk data is proportional to chunk_size ** 2.
##
## search_radius: This parameter determines the size of the search window around each pit.
## The search window is a square with side length 2 * search_radius + 1. Therefore, the memory
## required for storing the cost, previous row, and previous column arrays is proportional to
## max_pits * (2 * search_radius + 1) ** 2.
##
## max_pits: This parameter determines the maximum number of pits that can be processed in a single call
## to the kernel. More pits will require more memory for storing the cost, previous row, and previous column
## arrays, but will also allow more efficient use of the GPU by processing more pits in parallel.
##
## Data Movement
## The data in this module moves from the host (CPU memory) to the device (GPU memory) in several steps:
##
## 1. The input DEM is read into host memory using GDAL.
##
## 2. The DEM is divided into chunks, and each chunk is processed separately. For each chunk, the pits in the
## chunk are identified and their locations are stored in a NumPy array. The costs array, previous row array,
## and previous column array are initialized and sent to the device to store the results of the least cost path
## algorithm.
##
## 3. The pits array and the chunk data are copied to device memory using cuda.to_device.
##
## 4. The breach_pits_in_chunk_cuda function is launched on the GPU. This function performs the least cost
## path algorithm to breach the pits in the chunk. The results are stored in device memory.
##
## 5. After all pits in a chunk have been processed, the results are copied back to host memory using
## d_dem.copy_to_host(). The breached DEM for the chunk is then written to the output raster.
##

import math
from numba import cuda
import numpy as np
from osgeo import gdal
from overflow.util.constants import (
    DEFAULT_SEARCH_RADIUS,
    DEFAULT_MAX_PITS,
    UNVISITED_INDEX,
    EPSILON_GRADIENT,
    DEFAULT_CHUNK_SIZE,
    NEIGHBOR_OFFSETS,
)
from overflow.util.raster import raster_chunker, create_dataset
from overflow.breach_single_cell_pits import breach_single_cell_pits_in_chunk


@cuda.jit(device=True)
def push_heap(
    heap_costs: np.ndarray,
    heap_rows: np.ndarray,
    heap_cols: np.ndarray,
    heap_index: int,
    heap_size: int,
    new_cost: float,
    new_row: int,
    new_col: int,
) -> None:
    """
    Inserts a new element into a min-heap structure represented by arrays,
    and performs a 'bubble-up' operation to maintain the heap property.

    Parameters:
    - heap_costs (numpy.ndarray): Array representing the costs associated with each element in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_rows (numpy.ndarray): Array representing the row indices associated with each element in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_cols (numpy.ndarray): Array representing the column indices associated with each element in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_index (int): Index of the heap to modify in the heap arrays.
    - heap_size (int): Current size of the heap at index heap_index.
    - new_cost (float): Cost associated with the new element.
    - new_row (int): Row index associated with the new element.
    - new_col (int): Column index associated with the new element.

    Returns:
    - None: The function modifies the heap arrays in-place.
    """
    # Check if the heap is full
    if heap_size >= heap_costs[heap_index].shape[0]:
        pass  # do not overflow the heap
    else:
        # Insert the new element at the end of the heap
        heap_costs[heap_index, heap_size] = new_cost
        heap_rows[heap_index, heap_size] = new_row
        heap_cols[heap_index, heap_size] = new_col

        # Start the bubble-up operation from the last element
        child_index = heap_size
        parent_index = (child_index - 1) // 2

        # Continue the bubble-up operation until the root of the heap is reached
        # or the parent's cost is less than the child's cost
        while (
            child_index > 0
            and heap_costs[heap_index, child_index]
            < heap_costs[heap_index, parent_index]
        ):
            # Swap the child and parent costs
            (
                heap_costs[heap_index, child_index],
                heap_costs[heap_index, parent_index],
            ) = (
                heap_costs[heap_index, parent_index],
                heap_costs[heap_index, child_index],
            )
            # Swap the child and parent rows
            heap_rows[heap_index, child_index], heap_rows[heap_index, parent_index] = (
                heap_rows[heap_index, parent_index],
                heap_rows[heap_index, child_index],
            )
            # Swap the child and parent columns
            heap_cols[heap_index, child_index], heap_cols[heap_index, parent_index] = (
                heap_cols[heap_index, parent_index],
                heap_cols[heap_index, child_index],
            )
            # Move up to the next level of the heap
            child_index = parent_index
            parent_index = (child_index - 1) // 2


@cuda.jit(device=True)
def pop_heap(
    heap_costs: np.ndarray,
    heap_rows: np.ndarray,
    heap_cols: np.ndarray,
    heap_index: int,
    heap_size: int,
) -> tuple[float, int, int]:
    """
    Pop the smallest element from the priority queue represented by a binary heap and
    restore the heap property by performing a 'bubble down' operation.

    Parameters:
    - heap_costs (np.ndarray): 2D array representing the costs of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_rows (np.ndarray): 2D array representing the row indices of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_cols (np.ndarray): 2D array representing the column indices of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_index (int): Index of the heap to modify in the heap arrays.
    - heap_size (int): Current size of the heap at index heap_index.

    Returns:
     - tuple[float, int, int]: A tuple containing the cost, row, and column indices of the
        popped element.
    """
    # Pop the smallest element from the priority queue
    popped_cost = heap_costs[heap_index, 0]
    popped_row = heap_rows[heap_index, 0]
    popped_col = heap_cols[heap_index, 0]

    # Move the last element in the heap to the root
    heap_costs[heap_index, 0] = heap_costs[heap_index, heap_size - 1]
    heap_rows[heap_index, 0] = heap_rows[heap_index, heap_size - 1]
    heap_cols[heap_index, 0] = heap_cols[heap_index, heap_size - 1]

    # Bubble down
    parent_index = 0
    while True:
        child_index1 = parent_index * 2 + 1
        child_index2 = parent_index * 2 + 2

        # If the first child exists and is smaller than the parent (and the second child if it exists)
        if (
            child_index1 < heap_size - 1
            and (
                child_index2 >= heap_size - 1
                or heap_costs[heap_index, child_index1]
                < heap_costs[heap_index, child_index2]
            )
            and heap_costs[heap_index, child_index1]
            < heap_costs[heap_index, parent_index]
        ):
            # Swap the first child and parent
            (
                heap_costs[heap_index, child_index1],
                heap_costs[heap_index, parent_index],
            ) = (
                heap_costs[heap_index, parent_index],
                heap_costs[heap_index, child_index1],
            )
            heap_rows[heap_index, child_index1], heap_rows[heap_index, parent_index] = (
                heap_rows[heap_index, parent_index],
                heap_rows[heap_index, child_index1],
            )
            heap_cols[heap_index, child_index1], heap_cols[heap_index, parent_index] = (
                heap_cols[heap_index, parent_index],
                heap_cols[heap_index, child_index1],
            )
            parent_index = child_index1
        # If the second child exists and is smaller than the parent
        elif (
            child_index2 < heap_size - 1
            and heap_costs[heap_index, child_index2]
            < heap_costs[heap_index, parent_index]
        ):
            # Swap the second child and parent
            (
                heap_costs[heap_index, child_index2],
                heap_costs[heap_index, parent_index],
            ) = (
                heap_costs[heap_index, parent_index],
                heap_costs[heap_index, child_index2],
            )
            heap_rows[heap_index, child_index2], heap_rows[heap_index, parent_index] = (
                heap_rows[heap_index, parent_index],
                heap_rows[heap_index, child_index2],
            )
            heap_cols[heap_index, child_index2], heap_cols[heap_index, parent_index] = (
                heap_cols[heap_index, parent_index],
                heap_cols[heap_index, child_index2],
            )
            parent_index = child_index2
        else:
            break

    return popped_cost, popped_row, popped_col


@cuda.jit(device=True)
def process_neighbor(
    heap_index: int,
    neighbor_row: int,
    neighbor_col: int,
    row_offset: int,
    col_offset: int,
    current_cell_cost: float,
    initial_elevation: float,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    cost_multiplier: float,
    current_row: int,
    current_col: int,
    heap_size: int,
    heap_costs: np.ndarray,
    heap_rows: np.ndarray,
    heap_cols: np.ndarray,
) -> int:
    """
    Process a neighboring cell in a Digital Elevation Model (DEM) grid for pathfinding.

    This function calculates the cost of moving to a neighboring cell and updates the
    cost and previous cell information if the calculated cost is lower than the current
    cost of the neighbor. It also enqueues the neighbor into a priority queue if the
    cost is updated and returns the new heap size after the operation completes.

    Parameters:
    - heap_index (int): Index of the heap to modify in the heap arrays.
    - neighbor_row (int): Row index of the neighboring cell.
    - neighbor_col (int): Column index of the neighboring cell.
    - row_offset (int): Offset of the row index of the neighboring cell in the search window.
    - col_offset (int): Offset of the column index of the neighboring cell in the search window.
    - current_cell_cost (float): Current cost of reaching the current cell.
    - initial_elevation (float): Initial elevation of the path.
    - dem (np.ndarray): 2D array representing the Digital Elevation Model grid.
    - dem_no_data_value (float): Value indicating no data in the DEM grid.
    - costs_array (np.ndarray): 3D array storing the accumulated costs of reaching
      each cell in the grid.
    - prev_rows_array (np.ndarray): 3D array storing the row indices of the previous
      cells for each cell in the grid.
    - prev_cols_array (np.ndarray): 3D array storing the column indices of the previous
      cells for each cell in the grid.
    - cost_multiplier (float): Multiplier factor for cost calculation considering diagonal movement.
    - current_row (int): Row index of the current cell.
    - current_col (int): Column index of the current cell.
    - heap_size (int): Current size of the heap at index heap_index.
    - heap_costs (np.ndarray): 2D array representing the costs of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_rows (np.ndarray): 2D array representing the row indices of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.
    - heap_cols (np.ndarray): 2D array representing the column indices of elements in the heap.
        This is a (n, m) array, where n is the number of heaps and m is the maximum allowed size of the heap.

    Returns:
     - int: Updated size of the priority queue after potentially enqueueing the neighbor.
    """
    # Get the elevation of the neighboring cell
    neighbor_elevation = dem[neighbor_row, neighbor_col]

    # Check if the neighboring cell has valid data
    if neighbor_elevation == dem_no_data_value or math.isnan(neighbor_elevation):
        neighbor_elevation = -np.inf

    # Calculate the cost of moving to the neighboring cell
    if neighbor_elevation != -np.inf:
        neighbor_cost = current_cell_cost + cost_multiplier * (
            neighbor_elevation - initial_elevation
        )
    else:
        neighbor_cost = current_cell_cost

    # Check if the calculated cost is less than the current cost of the neighbor
    if (
        neighbor_cost
        < costs_array[heap_index, neighbor_row + row_offset, neighbor_col + col_offset]
    ):
        # If so, update the cost and previous cell information of the neighbor
        costs_array[
            heap_index, neighbor_row + row_offset, neighbor_col + col_offset
        ] = neighbor_cost
        prev_rows_array[
            heap_index, neighbor_row + row_offset, neighbor_col + col_offset
        ] = current_row
        prev_cols_array[
            heap_index, neighbor_row + row_offset, neighbor_col + col_offset
        ] = current_col

        # Enqueue the neighbor into the priority queue
        push_heap(
            heap_costs,
            heap_rows,
            heap_cols,
            heap_index,
            heap_size,
            neighbor_cost,
            neighbor_row,
            neighbor_col,
        )

        # Increase the size of the heap as we have added a new element
        return heap_size + 1

    # If the calculated cost is not less, return the current heap size
    return heap_size


@cuda.jit
def breach_dem(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    heap_costs: np.ndarray,
    heap_rows: np.ndarray,
    heap_cols: np.ndarray,
    breach_points_found: np.ndarray,
    current_row: np.ndarray,
    current_col: np.ndarray,
    init_elevation: np.ndarray,
    row_offset: np.ndarray,
    col_offset: np.ndarray,
    search_radius: int,
    max_cost: float,
):
    """
    Find breach points in a Digital Elevation Model (DEM) using Dijkstra's algorithm.

    This function iteratively explores neighboring cells around pit locations (depressions
    in the DEM) to identify breach points, where water can escape the depression.
    It uses Dijkstra's algorithm to calculate the cost of moving to neighboring cells
    based on the elevation change. The function continues exploration until it finds a
    breach point or covers the search window.

    Parameters:
    - pits (np.ndarray): 2D array representing the coordinates of pit locations (row, col).
    - dem (np.ndarray): 2D array representing the Digital Elevation Model grid.
    - dem_no_data_value (float): Value indicating no data in the DEM grid.
    - costs_array (np.ndarray): 3D array storing the accumulated costs of reaching each
        cell in the grid.
    - prev_rows_array (np.ndarray): 3D array storing the row indices of the previous cells
        for each cell in the grid.
    - prev_cols_array (np.ndarray): 3D array storing the column indices of the previous
        cells for each cell in the grid.
    - heap_costs (np.ndarray): 2D array storing the costs of elements in each priority queue.
        There is one priority queue for each pit location.
    - heap_rows (np.ndarray): 2D array storing the row indices of elements in each priority
        queue. There is one priority queue for each pit location.
    - heap_cols (np.ndarray): 2D array storing the column indices of elements in each priority
        queue. There is one priority queue for each pit location.
    - breach_points_found (np.ndarray): 1D array indicating whether breach points have been
        found for each pit location.
    - current_row (np.ndarray): 1D array storing the current row index for each pit location.
    - current_col (np.ndarray): 1D array storing the current column index for each pit location.
    - init_elevation (np.ndarray): 1D array storing the initial elevation for each pit location.
    - row_offset (np.ndarray): 1D array storing the row offset for each pit location within
        the search window.
    - col_offset (np.ndarray): 1D array storing the column offset for each pit location within
        the search window.
    - search_radius (int): Radius of the search window around each pit location.

    Returns:
    - None: The function modifies the input arrays in place
    """
    # pylint doesn't recognize the cuda.grid() function
    # pylint: disable=no-value-for-parameter
    i = cuda.grid(1)  # Get the current thread index
    # If the thread index is out of bounds, return
    if i >= pits.shape[0]:
        return
    # Define the size of the search window
    search_window_size = 2 * search_radius + 1
    # Define the neighboring cells to consider in constant global memory
    neighbors = cuda.const.array_like(NEIGHBOR_OFFSETS)
    # Initialize the current row, column, and cost for the current pit location
    current_row[i] = pits[i, 0]
    current_col[i] = pits[i, 1]
    current_cost = 0.0
    init_elevation[i] = dem[current_row[i], current_col[i]]
    row_offset[i] = search_radius - current_row[i]
    col_offset[i] = search_radius - current_col[i]
    costs_array[i, current_row[i] + row_offset[i], current_col[i] + col_offset[i]] = (
        current_cost
    )
    # Initialize the breach point found flag for the current pit location
    breach_points_found[i] = False
    # Initialize the heap size
    heap_size = 0
    # Push the current cell into the heap
    push_heap(
        heap_costs,
        heap_rows,
        heap_cols,
        i,
        heap_size,
        current_cost,
        current_row[i],
        current_col[i],
    )
    # Increase the heap size
    heap_size += 1
    # Start the main loop of Dijkstra's algorithm
    while 0 < heap_size < search_window_size**2:
        # Pop the cell with the lowest cost from the heap
        current_cost, current_row[i], current_col[i] = pop_heap(
            heap_costs, heap_rows, heap_cols, i, heap_size
        )
        # Decrease the heap size
        heap_size -= 1
        # Check if the current cell is a breach point
        if (
            dem[current_row[i], current_col[i]] < init_elevation[i]
            or dem[current_row[i], current_col[i]] == dem_no_data_value
            or math.isnan(dem[current_row[i], current_col[i]])
        ):
            # If so, set the breach point found flag and break the loop
            breach_points_found[i] = True
            break
        # Process all neighboring cells
        for dr, dc in neighbors:
            # Calculate the row and column indices of the neighboring cell
            next_row, next_col = current_row[i] + dr, current_col[i] + dc
            # Compute the cost multiplier for diagonal movement
            multiplier = 1 if dr == 0 or dc == 0 else math.sqrt(2)
            # Check if the neighboring cell is in bounds
            is_in_bounds = (
                # check if the cell is inside the DEM
                0 <= next_row < dem.shape[0]
                and 0 <= next_col < dem.shape[1]
                # check if the cell is inside the search window
                and 0 <= next_row + row_offset[i] < search_window_size
                and 0 <= next_col + col_offset[i] < search_window_size
            )
            # If the neighboring cell is in bounds, process it
            if is_in_bounds and current_cost < max_cost:
                heap_size = process_neighbor(
                    i,
                    next_row,
                    next_col,
                    row_offset[i],
                    col_offset[i],
                    current_cost,
                    init_elevation[i],
                    dem,
                    dem_no_data_value,
                    costs_array,
                    prev_rows_array,
                    prev_cols_array,
                    multiplier,
                    current_row[i],
                    current_col[i],
                    heap_size,
                    heap_costs,
                    heap_rows,
                    heap_cols,
                )


@cuda.jit
def reconstruct_path(
    breach_points_rows: np.ndarray,
    breach_points_cols: np.ndarray,
    initial_elevation: np.ndarray,
    dem: np.ndarray,
    no_data_value: float,
    previous_cell_rows: np.ndarray,
    previous_cell_cols: np.ndarray,
    row_offset: np.ndarray,
    col_offset: np.ndarray,
    breach_points_found: np.ndarray,
):
    """
    Reconstruct the least-cost path from the final breach point to the pit cell.

    This function reconstructs the least-cost path from the final breach point back to the pit cell using the
    information stored in the previous cell arrays. It applies gradient to the DEM to create the breach path.

    Parameters:
        breach_points_rows (np.ndarray): Row indices of the breach points being processed.
        breach_points_cols (np.ndarray): Column indices of the breach points.
        initial_elevation (np.ndarray): Initial elevations of the pit cells.
        dem (np.ndarray): Digital Elevation Model (DEM) array.
        no_data_value (float): Value indicating no data in the DEM grid.
        previous_cell_rows (np.ndarray): 3D array storing the previous row indices of the path.
        previous_cell_cols (np.ndarray): 3D array storing the previous column indices of the path.
        row_offset (np.ndarray): Offset of the row index of the pit in the search window.
        col_offset (np.ndarray): Offset of the column index of the pit in the search window.
        breach_points_found (np.ndarray): Flag indicating if a breach point was found for each pit.

    Returns:
        None, the function modifies the dem in place with atomic operations.
    """
    # pylint doesn't recognize the cuda.grid() function
    # pylint: disable=no-value-for-parameter
    thread_index = cuda.grid(1)  # Get the current thread index
    # If the thread index is out of bounds or no breach point was found, return
    if (
        thread_index >= breach_points_rows.shape[0]
        or not breach_points_found[thread_index]
    ):
        return
    # Get the row and column indices of the final breach point
    breach_point_row, breach_point_col = (
        breach_points_rows[thread_index],
        breach_points_cols[thread_index],
    )

    # Get the elevation of the final breach point
    final_elevation = dem[breach_point_row, breach_point_col]

    # If the final elevation is a no data value, set it to negative infinity
    final_elevation = final_elevation if final_elevation != no_data_value else -np.inf

    # Initialize the path length
    path_length = 0

    # Initialize the backlink row and column indices
    backlink_row, backlink_col = breach_point_row, breach_point_col

    # Traverse the path from the final breach point to the pit cell
    while UNVISITED_INDEX not in (backlink_row, backlink_col):
        # Increase the path length
        path_length += 1

        # Update the backlink row and column indices
        backlink_row, backlink_col = (
            previous_cell_rows[
                thread_index,
                backlink_row + row_offset[thread_index],
                backlink_col + col_offset[thread_index],
            ],
            previous_cell_cols[
                thread_index,
                backlink_row + row_offset[thread_index],
                backlink_col + col_offset[thread_index],
            ],
        )

    # Exclude the pit cell from the path length
    path_length -= 1

    # Apply gradient to the DEM to create the breach path
    for step in range(path_length):
        # If we're breaching to a no data cell, don't modify the first cell
        if final_elevation == -np.inf and step > 0:
            # Assume a small gradient
            new_elevation = (
                initial_elevation[thread_index]
                - (path_length - step) * EPSILON_GRADIENT
            )

            # Update the elevation of the current cell to the new elevation
            # pylint doesn't recognize the cuda.atomic.min() function
            # pylint: disable=too-many-function-args
            cuda.atomic.min(dem, (breach_point_row, breach_point_col), new_elevation)
        else:
            # don't apply gradient in flat areas
            if (
                dem[breach_point_row, breach_point_col]
                == initial_elevation[thread_index]
            ):
                continue
            # Calculate the new elevation based on the gradient
            new_elevation = (
                final_elevation
                + (initial_elevation[thread_index] - final_elevation)
                * step
                / path_length
            )

            # Update the elevation of the current cell to the new elevation
            # pylint doesn't recognize the cuda.atomic.min() function
            # pylint: disable=too-many-function-args
            cuda.atomic.min(dem, (breach_point_row, breach_point_col), new_elevation)

        # Move to the previous cell in the path
        breach_point_row, breach_point_col = (
            previous_cell_rows[
                thread_index,
                breach_point_row + row_offset[thread_index],
                breach_point_col + col_offset[thread_index],
            ],
            previous_cell_cols[
                thread_index,
                breach_point_row + row_offset[thread_index],
                breach_point_col + col_offset[thread_index],
            ],
        )


@cuda.jit
def clear_arrays(
    cost_arrays: np.ndarray,
    previous_row_arrays: np.ndarray,
    previous_column_arrays: np.ndarray,
    current_rows: np.ndarray,
    current_columns: np.ndarray,
    initial_elevations: np.ndarray,
    breach_points_found_flags: np.ndarray,
    row_offsets: np.ndarray,
    column_offsets: np.ndarray,
):
    """
    Clear the arrays used in the least-cost path calculation.

    This function is used to reset the arrays used in the least-cost path calculation at the start of each iteration.
    It sets the cost arrays to infinity, the previous cell arrays to a value indicating an unvisited cell, the current
    cell arrays to a value indicating an unvisited cell, the initial elevation array to NaN, the breach points found
    flags to False, and the row and column offsets to a value indicating an unvisited cell.

    Parameters:
        cost_arrays (np.ndarray): The arrays storing the cost of each cell in the path.
        previous_row_arrays (np.ndarray): The arrays storing the row indices of the previous cells in the path.
        previous_column_arrays (np.ndarray): The arrays storing the column indices of the previous cells in the path.
        current_rows (np.ndarray): The array storing the current row indices.
        current_columns (np.ndarray): The array storing the current column indices.
        initial_elevations (np.ndarray): The array storing the initial elevations of the pit cells.
        breach_points_found_flags (np.ndarray): The flags indicating if a breach point was found for each pit.
        row_offsets (np.ndarray): The array storing the row offsets of the final breach points in the search window.
        column_offsets (np.ndarray): The array storing the column offsets of the final breach points in the search
        window.

    Returns:
        None
    """
    # pylint doesn't recognize the cuda.grid() function
    # pylint: disable=no-value-for-parameter
    thread_index = cuda.grid(1)  # Get the current thread index

    # If the thread index is out of bounds, return
    if thread_index >= cost_arrays.shape[0]:
        return

    # Set the cost array to infinity
    cost_arrays[thread_index].fill(np.inf)

    # Set the previous cell arrays to a value indicating an unvisited cell
    previous_row_arrays[thread_index].fill(UNVISITED_INDEX)
    previous_column_arrays[thread_index].fill(UNVISITED_INDEX)

    # Set the current cell arrays to a value indicating an unvisited cell
    current_rows[thread_index] = UNVISITED_INDEX
    current_columns[thread_index] = UNVISITED_INDEX

    # Set the initial elevation to NaN
    initial_elevations[thread_index] = np.nan

    # Set the breach points found flag to False
    breach_points_found_flags[thread_index] = False

    # Set the row and column offsets to a value indicating an unvisited cell
    row_offsets[thread_index] = UNVISITED_INDEX
    column_offsets[thread_index] = UNVISITED_INDEX


def breach_pits_in_chunk_cuda(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    d_costs_array,
    d_prev_rows_array,
    d_prev_cols_array,
    d_heap_costs,
    d_heap_rows,
    d_heap_cols,
    d_breach_points_found,
    d_current_row,
    d_current_col,
    d_init_elevation,
    d_row_offset,
    d_col_offset,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_pits: int = DEFAULT_MAX_PITS,
    max_cost: float = np.inf,
) -> np.ndarray:
    """
    Breach pits in a chunk of a Digital Elevation Model (DEM) using CUDA.

    This function breaches pits in a chunk of a DEM using a CUDA kernel. It divides the pits into chunks of a maximum
    size, copies the pits and the DEM to the device, launches the CUDA kernel to breach the DEM, reconstructs the
    least-cost path from the final breach point to the pit cell, and clears the arrays used in the calculation.

    Parameters:
        pits (np.ndarray): Array of pit indices in the DEM.
        dem (np.ndarray): The DEM array.
        dem_no_data_value (float): Value indicating no data in the DEM grid.
        d_costs_array: Device array storing the cost of each cell in the path.
        d_prev_rows_array: Device array storing the row indices of the previous cells in the path.
        d_prev_cols_array: Device array storing the column indices of the previous cells in the path.
        d_heap_costs: Device array used as a heap to store the costs of the cells.
        d_heap_rows: Device array used as a heap to store the row indices of the cells.
        d_heap_cols: Device array used as a heap to store the column indices of the cells.
        d_breach_points_found: Device array storing flags indicating if a breach point was found for each pit.
        d_current_row: Device array storing the current row indices.
        d_current_col: Device array storing the current column indices.
        d_init_elevation: Device array storing the initial elevations of the pit cells.
        d_row_offset: Device array storing the row offsets of the final breach points in the search window.
        d_col_offset: Device array storing the column offsets of the final breach points in the search window.
        search_radius (int, optional): The radius of the search window around each pit. Defaults to
            DEFAULT_SEARCH_RADIUS.
        max_pits (int, optional): The maximum number of pits to process in each chunk. Defaults to DEFAULT_MAX_PITS.

    Returns:
        np.ndarray: The breached DEM.
    """
    d_dem = cuda.to_device(dem)
    for i in range(0, pits.shape[0], max_pits):
        chunk_pits = np.ascontiguousarray(pits[i : i + max_pits])
        # Copy arrays to device
        d_pits = cuda.to_device(chunk_pits)
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (max_pits + (threads_per_block - 1)) // threads_per_block
        breach_dem[blocks_per_grid, threads_per_block](
            d_pits,
            d_dem,
            dem_no_data_value,
            d_costs_array,
            d_prev_rows_array,
            d_prev_cols_array,
            d_heap_costs,
            d_heap_rows,
            d_heap_cols,
            d_breach_points_found,
            d_current_row,
            d_current_col,
            d_init_elevation,
            d_row_offset,
            d_col_offset,
            search_radius,
            max_cost,
        )
        # reconstruct paths
        reconstruct_path[blocks_per_grid, threads_per_block](
            d_current_row,
            d_current_col,
            d_init_elevation,
            d_dem,
            dem_no_data_value,
            d_prev_rows_array,
            d_prev_cols_array,
            d_row_offset,
            d_col_offset,
            d_breach_points_found,
        )
        # clear arrays on device
        clear_arrays[blocks_per_grid, threads_per_block](
            d_costs_array,
            d_prev_rows_array,
            d_prev_cols_array,
            d_current_row,
            d_current_col,
            d_init_elevation,
            d_breach_points_found,
            d_row_offset,
            d_col_offset,
        )
    return d_dem.copy_to_host()


def breach_paths_least_cost_cuda(
    input_dem_path: str,
    output_dem_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_pits: int = DEFAULT_MAX_PITS,
    max_cost: float = np.inf,
):
    """
    Breach paths in a Digital Elevation Model (DEM) using CUDA.

    This function breaches paths in a DEM using a CUDA kernel. It reads the input DEM, creates an output DEM with the
    same properties, initializes arrays used in the calculation, copies the arrays to the device, processes the DEM in
    chunks, breaches the pits in each chunk, and writes the breached DEM to the output file.

    Parameters:
        input_dem_path (str): Path to the input DEM file.
        output_dem_path (str): Path to the output DEM file.
        chunk_size (int, optional): The size of the chunks in which the DEM is processed.
        Defaults to DEFAULT_CHUNK_SIZE.
        search_radius (int, optional): The radius of the search window around each pit.
        Defaults to DEFAULT_SEARCH_RADIUS.
        max_pits (int, optional): The maximum number of pits to process in each chunk.
        Defaults to DEFAULT_MAX_PITS.

    Returns:
        None
    """
    # Open the input DEM
    input_ds = gdal.Open(input_dem_path)
    projection = input_ds.GetProjection()
    geotransform = input_ds.GetGeoTransform()
    input_band = input_ds.GetRasterBand(1)
    input_no_data_value = input_band.GetNoDataValue()

    # Create the output DEM
    output_ds = create_dataset(
        output_dem_path,
        input_no_data_value,
        gdal.GDT_Float32,
        input_ds.RasterXSize,
        input_ds.RasterYSize,
        geotransform,
        projection,
    )
    output_band = output_ds.GetRasterBand(1)

    # Allocate memory for arrays used in the calculation
    search_window_size = 2 * search_radius + 1
    costs_array = np.full(
        (max_pits, search_window_size, search_window_size), np.inf, dtype=np.float32
    )
    prev_rows_array = np.full(
        (max_pits, search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int32,
    )
    prev_cols_array = np.full(
        (max_pits, search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int32,
    )
    heap_costs = np.full((max_pits, search_window_size**2), np.inf, dtype=np.float32)
    heap_rows = np.full(
        (max_pits, search_window_size**2), UNVISITED_INDEX, dtype=np.int32
    )
    heap_cols = np.full(
        (max_pits, search_window_size**2), UNVISITED_INDEX, dtype=np.int32
    )
    breach_points_found = np.full(max_pits, False, dtype=np.bool_)
    current_row = np.full(max_pits, UNVISITED_INDEX, dtype=np.int32)
    current_col = np.full(max_pits, UNVISITED_INDEX, dtype=np.int32)
    init_elevation = np.full(max_pits, np.nan, dtype=np.float32)
    row_offset = np.full(max_pits, UNVISITED_INDEX, dtype=np.int32)
    col_offset = np.full(max_pits, UNVISITED_INDEX, dtype=np.int32)

    # Copy arrays to device
    d_costs_array = cuda.to_device(costs_array)
    d_prev_rows_array = cuda.to_device(prev_rows_array)
    d_prev_cols_array = cuda.to_device(prev_cols_array)
    d_heap_costs = cuda.to_device(heap_costs)
    d_heap_rows = cuda.to_device(heap_rows)
    d_heap_cols = cuda.to_device(heap_cols)
    d_breach_points_found = cuda.to_device(breach_points_found)
    d_current_row = cuda.to_device(current_row)
    d_current_col = cuda.to_device(current_col)
    d_init_elevation = cuda.to_device(init_elevation)
    d_row_offset = cuda.to_device(row_offset)
    d_col_offset = cuda.to_device(col_offset)

    # Process the DEM in chunks
    for chunk in raster_chunker(input_band, chunk_size, search_radius):
        # Breach the pits in the chunk
        pits_raster = breach_single_cell_pits_in_chunk(chunk.data, input_no_data_value)
        pits_array = np.argwhere(pits_raster)
        chunk.from_numpy(
            breach_pits_in_chunk_cuda(
                pits_array,
                chunk.data,
                input_no_data_value,
                d_costs_array,
                d_prev_rows_array,
                d_prev_cols_array,
                d_heap_costs,
                d_heap_rows,
                d_heap_cols,
                d_breach_points_found,
                d_current_row,
                d_current_col,
                d_init_elevation,
                d_row_offset,
                d_col_offset,
                search_radius,
                max_pits,
                max_cost,
            )
        )
        # Write the breached chunk to the output DEM
        chunk.write(output_band)

    # Close the datasets
    input_ds = None
    output_ds = None
