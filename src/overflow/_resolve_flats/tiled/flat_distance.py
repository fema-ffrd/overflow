import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from numba.types import int32

from overflow._util.perimeter import Int64Perimeter, get_tile_perimeter
from overflow._util.queue import Int64PairQueue as Queue
from overflow._util.raster import neighbor_generator

UNREACHABLE = np.iinfo(np.int32).max


@njit
def path_exists(
    labels_array: np.ndarray, row1: int, col1: int, row2: int, col2: int
) -> bool:
    """
    Check if a path exists between two points with the same label.

    This is a quick heuristic to check if a chebyshev path exists between two points with the same label.
    There may still be a path even if this function returns False, but if it returns True, then
    there is definitely a path and we can proceed with the Chebyshev distance calculation instead of
    the more complex BFS algorithm.

    Note that if the labels are different, this function will return True so that the BFS is not performed.

    This optimizes the performance of the algorithm in large open flat regions.
    """
    path = []
    current_row, current_col = row1, col1
    start_label = labels_array[row1, col1]
    end_label = labels_array[row2, col2]
    if start_label == 0 or end_label == 0:
        return True
    if start_label != end_label:
        return True
    while (current_row, current_col) != (row2, col2):
        path.append((current_row, current_col))
        if current_row < row2:
            current_row += 1
        elif current_row > row2:
            current_row -= 1
        if current_col < col2:
            current_col += 1
        elif current_col > col2:
            current_col -= 1
        if labels_array[current_row, current_col] != start_label:
            return False
    return True


@njit
def chebyshev_distance(
    labels_array: np.ndarray, row1: int, col1: int, row2: int, col2: int
) -> int:
    """
    Compute the Chebyshev distance between two points.

    The Chebyshev distance is the maximum of the horizontal and vertical distances between two
    points.

    Args:
        row1 (int): The row of the first point.
        col1 (int): The column of the first point.
        row2 (int): The row of the second point.
        col2 (int): The column of the second point.

    Returns:
        int: The Chebyshev distance between the two points.
    """
    # is path does not exist, return 0
    if not path_exists(labels_array, row1, col1, row2, col2):
        return int32(0)  # type: ignore[no-any-return]
    return int32(max(abs(row1 - row2), abs(col1 - col2)))  # type: ignore[no-any-return]


@njit(parallel=True)
def compute_min_dist_chebyshev_paths(
    labels_array: np.ndarray, labels_perimeter: Int64Perimeter, perimeter_count: int
) -> np.ndarray:
    """
    Compute the minimum distances between perimeter cells with the same label using Chebyshev paths.

    This function computes the minimum distances between perimeter cells with the same label using
    Chebyshev paths. It uses the Chebyshev distance between two points as a heuristic to determine
    if a path exists between two points with the same label. If a path exists, the Chebyshev distance
    is calculated and stored in the output array.

    """
    min_dist = np.zeros((perimeter_count, perimeter_count), dtype=np.int32)
    for i in prange(perimeter_count):
        label = labels_perimeter.data[i]
        if label == 0:
            continue
        from_row, from_col = labels_perimeter.get_row_col(i)
        for j in range(i + 1, perimeter_count):
            if labels_perimeter.data[j] == label:
                to_row, to_col = labels_perimeter.get_row_col(j)
                dist = chebyshev_distance(
                    labels_array, from_row, from_col, to_row, to_col
                )
                min_dist[i][j] = dist
                min_dist[j][i] = dist
    return min_dist


@njit(parallel=True)
def compute_min_dist_bfs(labels_array: np.ndarray) -> np.ndarray:
    """
    Compute the minimum distances between perimeter cells with the same label.

    This function takes a 2D array of labels and computes the minimum distances between perimeter
    cells that have the same label. It uses a breadth-first search (BFS) algorithm to traverse
    the cells with the same label and calculate the distances.

    Args:
        labels_array (np.ndarray): A 2D array of labels, where each cell is assigned a label value.
            Cells with the same label belong to the same flat region.

    Returns:
        np.ndarray: A 2D array of shape (perimeter_count, perimeter_count) representing the
            minimum distances between perimeter cells with the same label. The value at index
            (i, j) represents the minimum distance between perimeter cell i and perimeter cell j.
    """
    labels_perimeter = Int64Perimeter(
        get_tile_perimeter(labels_array),
        labels_array.shape[0],
        labels_array.shape[1],
        0,
    )
    perimeter_count = labels_perimeter.size()
    min_dist = compute_min_dist_chebyshev_paths(
        labels_array, labels_perimeter, perimeter_count
    )
    # min_dist = np.zeros((perimeter_count, perimeter_count), dtype=np.int64)
    for i in prange(perimeter_count):
        label = labels_perimeter.data[i]
        if label == 0:
            continue
        # check if all destination cells of the same label already have a min distance
        all_destinations_have_min_dist = True
        for j in range(perimeter_count):
            if i != j and labels_perimeter.data[j] == label and min_dist[i][j] == 0:
                all_destinations_have_min_dist = False
                break
        if all_destinations_have_min_dist:
            continue
        else:
            # set all distances to 0 for this cell
            for j in range(perimeter_count):
                if labels_perimeter.data[j] == label:
                    min_dist[i][j] = 0
                    min_dist[j][i] = 0
        visited = set()
        dist = int32(0)
        from_row, from_col = labels_perimeter.get_row_col(i)
        marker = (-1, -1)
        queue = Queue([(from_row, from_col)])
        queue.push(marker)
        while len(queue) > 1:
            row, col = queue.pop()
            if (row, col) == marker:
                dist += 1
                queue.push(marker)
                continue
            if (row, col) in visited:
                continue
            visited.add((row, col))
            if (
                row == 0
                or row == labels_array.shape[0] - 1
                or col == 0
                or col == labels_array.shape[1] - 1
            ):
                j = labels_perimeter.get_index(row, col)
                if min_dist[i][j] > 0:
                    continue
                min_dist[i][j] = dist
                min_dist[j][i] = dist
            for neighbor_row, neighbor_col in neighbor_generator(
                row, col, labels_array.shape[0], labels_array.shape[1]
            ):
                if labels_array[neighbor_row, neighbor_col] == label:
                    queue.push((neighbor_row, neighbor_col))

    return min_dist  # type: ignore[no-any-return]


@njit
def flood_fill_distance(
    labels_array, start, label, distances, perimeter_index, labels_perimeter
):
    rows, cols = labels_array.shape
    distances.fill(UNREACHABLE)  # Reset the distances array
    distances[start] = 0
    queue = Queue([(start[0], start[1])])
    # set of cells on the perimeter that need to be found
    remaining_cells = set()
    for i in range(perimeter_index, labels_perimeter.size()):
        if labels_perimeter.data[i] == label and perimeter_index != i:
            remaining_cells.add(i)
    while queue:
        x, y = queue.pop()
        current_dist = distances[x, y]

        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and labels_array[nx, ny] == label:
                new_dist = current_dist + 1  # All moves cost the same
                if new_dist < distances[nx, ny]:
                    distances[nx, ny] = new_dist
                    queue.push((nx, ny))
                    # if this is a cell on the perimeter, remove it from the set of remaining cells
                    if nx == 0 or nx == rows - 1 or ny == 0 or ny == cols - 1:
                        j = labels_perimeter.get_index(nx, ny)
                        if j in remaining_cells:
                            remaining_cells.remove(j)
        # if all remaining cells have been found, break the loop
        if not remaining_cells:
            break

    return distances


@njit
def compute_min_dist_flood(labels_array):
    labels_perimeter = Int64Perimeter(
        get_tile_perimeter(labels_array),
        labels_array.shape[0],
        labels_array.shape[1],
        0,
    )
    perimeter_count = labels_perimeter.size()
    # min_dist = np.full((perimeter_count, perimeter_count), 0, dtype=np.int32)
    min_dist = compute_min_dist_chebyshev_paths(
        labels_array, labels_perimeter, perimeter_count
    )

    # Pre-allocate a single distance array to be reused for all computations
    distances = np.full(labels_array.shape, UNREACHABLE, dtype=np.int32)

    for i in range(perimeter_count):
        label = labels_perimeter.data[i]
        if label == 0:
            continue
        # check if all destination cells of the same label already have a min distance
        all_destinations_have_min_dist = True
        for j in range(perimeter_count):
            if i != j and labels_perimeter.data[j] == label and min_dist[i][j] == 0:
                all_destinations_have_min_dist = False
                break
        if all_destinations_have_min_dist:
            continue

        from_pos = labels_perimeter.get_row_col(i)
        distances = flood_fill_distance(
            labels_array, from_pos, label, distances, i, labels_perimeter
        )

        for j in range(i + 1, perimeter_count):
            if labels_perimeter.data[j] == label:
                to_pos = labels_perimeter.get_row_col(j)
                min_dist[i, j] = distances[to_pos]
                min_dist[j, i] = distances[to_pos]

    return min_dist
