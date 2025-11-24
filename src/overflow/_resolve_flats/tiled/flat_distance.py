import numpy as np
from numba import njit  # type: ignore[attr-defined]
from numba.types import int32

from overflow._util.perimeter import Int64Perimeter, get_tile_perimeter

UNREACHABLE = np.iinfo(np.int32).max


@njit
def build_change_integral_images(labels: np.ndarray):
    """
    Build Integral Images (Summed Area Tables) for horizontal and vertical label changes.
    This allows O(1) checking if any sub-rectangle contains a mix of labels.
    """
    rows, cols = labels.shape

    # Identify boundaries (where label changes)
    # v_changes[i, j] = 1 if label[i,j] != label[i-1,j]
    # h_changes[i, j] = 1 if label[i,j] != label[i,j-1]
    v_changes = np.zeros((rows, cols), dtype=np.int32)
    h_changes = np.zeros((rows, cols), dtype=np.int32)

    for r in range(1, rows):
        for c in range(cols):
            if labels[r, c] != labels[r - 1, c]:
                v_changes[r, c] = 1

    for r in range(rows):
        for c in range(1, cols):
            if labels[r, c] != labels[r, c - 1]:
                h_changes[r, c] = 1

    # Compute Prefix Sums (Integral Images) in-place
    # Sum along rows
    for r in range(rows):
        for c in range(1, cols):
            v_changes[r, c] += v_changes[r, c - 1]
            h_changes[r, c] += h_changes[r, c - 1]

    # Sum along cols
    for r in range(1, rows):
        for c in range(cols):
            v_changes[r, c] += v_changes[r - 1, c]
            h_changes[r, c] += h_changes[r - 1, c]

    return v_changes, h_changes


@njit
def is_rect_homogeneous(
    r1: int, c1: int, r2: int, c2: int, v_sat: np.ndarray, h_sat: np.ndarray
) -> bool:
    """Check if bounding box contains mixed labels using O(1) lookup."""
    rmin, rmax = min(r1, r2), max(r1, r2)
    cmin, cmax = min(c1, c2), max(c1, c2)

    # Check Vertical Changes
    if rmax > rmin:
        A = v_sat[rmin, cmin - 1] if cmin > 0 else 0
        B = v_sat[rmin, cmax]
        C = v_sat[rmax, cmin - 1] if cmin > 0 else 0
        D = v_sat[rmax, cmax]
        if (D - B - C + A) > 0:
            return False

    # Check Horizontal Changes
    if cmax > cmin:
        rmin_minus = rmin - 1
        A = h_sat[rmin_minus, cmin] if rmin_minus >= 0 else 0
        B = h_sat[rmin_minus, cmax] if rmin_minus >= 0 else 0
        C = h_sat[rmax, cmin]
        D = h_sat[rmax, cmax]
        if (D - B - C + A) > 0:
            return False

    return True


@njit
def flood_fill_static(
    labels_array: np.ndarray,
    start_node: tuple[int, int],
    label: int,
    distances: np.ndarray,
    queue_row_buff: np.ndarray,
    queue_col_buff: np.ndarray,
):
    """
    BFS using pre-allocated static arrays.
    """
    rows, cols = labels_array.shape
    distances.fill(UNREACHABLE)

    start_r, start_c = start_node
    distances[start_r, start_c] = 0

    # Manual Queue pointers
    q_head = 0
    q_tail = 0

    # Push start
    queue_row_buff[q_tail] = start_r
    queue_col_buff[q_tail] = start_c
    q_tail += 1

    while q_head < q_tail:
        r = queue_row_buff[q_head]
        c = queue_col_buff[q_head]
        q_head += 1

        dist = distances[r, c]
        next_dist = dist + 1

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    if labels_array[nr, nc] == label:
                        if distances[nr, nc] == UNREACHABLE:
                            distances[nr, nc] = next_dist
                            queue_row_buff[q_tail] = nr
                            queue_col_buff[q_tail] = nc
                            q_tail += 1
    return distances


@njit
def compute_min_dist_flood(labels_array: np.ndarray) -> np.ndarray:
    """
    Compute minimum distances between perimeter cells.
    """
    perimeter_data = get_tile_perimeter(labels_array)
    rows, cols = labels_array.shape
    labels_perimeter = Int64Perimeter(perimeter_data, rows, cols, 0)
    perimeter_count = labels_perimeter.size()

    # Build Integral Images
    # This allows us to skip BFS for any pair of points that form a clean rectangle
    v_sat, h_sat = build_change_integral_images(labels_array)
    min_dist = np.zeros((perimeter_count, perimeter_count), dtype=np.int32)

    for i in range(perimeter_count):
        label = labels_perimeter.data[i]
        if label == 0:
            continue

        from_pos = labels_perimeter.get_row_col(i)

        # Attempt Integral Image Check for all neighbors
        for j in range(i + 1, perimeter_count):
            if labels_perimeter.data[j] == label:
                to_pos = labels_perimeter.get_row_col(j)

                if is_rect_homogeneous(
                    from_pos[0], from_pos[1], to_pos[0], to_pos[1], v_sat, h_sat
                ):
                    d = int32(
                        max(abs(from_pos[0] - to_pos[0]), abs(from_pos[1] - to_pos[1]))
                    )
                    min_dist[i, j] = d
                    min_dist[j, i] = d

        # Fallback Check
        # If we filled all neighbors using the rect check, we don't need BFS.
        needs_bfs = False
        for j in range(perimeter_count):
            if i != j and labels_perimeter.data[j] == label and min_dist[i][j] == 0:
                needs_bfs = True
                break

        if not needs_bfs:
            continue

        # BFS
        bfs_distances = np.empty((rows, cols), dtype=np.int32)
        q_row_buff = np.empty(rows * cols, dtype=np.int64)
        q_col_buff = np.empty(rows * cols, dtype=np.int64)

        flood_fill_static(
            labels_array, from_pos, label, bfs_distances, q_row_buff, q_col_buff
        )

        for j in range(i + 1, perimeter_count):
            if labels_perimeter.data[j] == label:
                to_pos = labels_perimeter.get_row_col(j)
                d = bfs_distances[to_pos[0], to_pos[1]]
                if d != UNREACHABLE:
                    min_dist[i, j] = d
                    min_dist[j, i] = d

    return min_dist
