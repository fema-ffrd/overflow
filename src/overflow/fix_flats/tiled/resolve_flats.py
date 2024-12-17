from numba import njit
from numba.typed import List  # pylint: disable=no-name-in-module
import numpy as np
from overflow.util.constants import FLOW_DIRECTION_UNDEFINED, FLOW_DIRECTION_NODATA
from overflow.util.raster import neighbor_generator
from overflow.fix_flats.core.fix_flats import label_flats
from overflow.util.queue import Int64PairQueue as Queue


@njit
def compute_gradient(
    labels: np.ndarray,
    fdr: np.ndarray,
    edges_cells: list,
) -> np.ndarray:
    """
    Compute the gradient from the given edge cells across the flats.

    This function uses a flood-fill approach to compute the gradient from the given edge cells
    across the flats. It assigns increasing values to cells as it moves away from the edge cells,
    creating a gradient that can be used to determine flow directions within the flats.

    Args:
        labels (np.ndarray): Array indicating which flat each cell is a member of. Cells in a flat
            have a value greater than zero indicating the label of the flat, otherwise, they have a
            value of 0.
        fdr (np.ndarray): Array containing flow directions for each cell. Cells without a local
            gradient have a value of FLOW_DIRECTION_UNDEFINED, while all other cells have defined
            flow directions or FLOW_DIRECTION_NODATA.
        edges_cells (list): List of edge cells from which to compute the gradient. These can be
            either high edge cells or low edge cells.

    Returns:
        np.ndarray: Array containing the computed gradient values. Cells not part of a flat have a
            value of 0.
    """
    graident = np.zeros_like(labels, dtype=np.int64)
    if len(edges_cells) == 0:
        return graident
    # Create a FIFO queue to store cells for processing
    edges_cells = Queue(edges_cells)
    loops = 1
    marker = (-1, -1)
    edges_cells.push(marker)
    while len(edges_cells) > 1:
        row, col = edges_cells.pop()
        # Check if the marker is encountered, indicating the end of a loop
        if row == marker[0] and col == marker[1]:
            loops += 1
            edges_cells.push(marker)
            continue
        # Skip cells that have already been assigned a gradient value
        if graident[row, col] > 0:
            continue
        # Assign the current loop value as the gradient value for the cell
        graident[row, col] = loops
        # Add neighboring cells within the same flat to the queue for processing
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                edges_cells.push((neighbor_row, neighbor_col))
    return graident


@njit
def flat_edges_tile(dem: np.ndarray, fdr: np.ndarray) -> tuple[list, list]:
    """
    Detects high and low edge cells in a tile. This is the tiled version of Algorithm 3 FlatEdges
    from https://rbarnes.org/sci/2014_flats.pdf

    Args:
        dem (np.ndarray): Digital Elevation Model array.
        fdr (np.ndarray): Flow direction array.

    Returns:
        Tuple[list, list]: Lists of high and low edge cells.
    """
    high_edges = List()
    low_edges = List()

    for row in range(1, dem.shape[0] - 1):
        for col in range(1, dem.shape[1] - 1):
            for neighbor_row, neighbor_col in neighbor_generator(
                row, col, fdr.shape[0], fdr.shape[1]
            ):
                # continue if the neighbor is nodata
                fdr_neighbor = fdr[neighbor_row, neighbor_col]
                if fdr_neighbor == FLOW_DIRECTION_NODATA:
                    continue
                fdr_current = fdr[row, col]
                if (
                    fdr_current != FLOW_DIRECTION_UNDEFINED
                    and (
                        fdr_neighbor == FLOW_DIRECTION_UNDEFINED
                        or fdr_neighbor == FLOW_DIRECTION_NODATA
                    )
                    and dem[row, col] == dem[neighbor_row, neighbor_col]
                ):
                    # cell is a low edge cell since it has a defined flow direction and a neighbor does not
                    # because the neighbor is a flat cell or nodata cell
                    low_edges.append((row - 1, col - 1))
                    break
                if (
                    fdr_current == FLOW_DIRECTION_UNDEFINED
                    and dem[row, col] < dem[neighbor_row, neighbor_col]
                ):
                    # cell is a high edge cell since it has no defined flow direction and a neighbor is higher
                    high_edges.append((row - 1, col - 1))
                    break
    return high_edges, low_edges


@njit
def compute_labels_and_gradients(
    dem: np.ndarray, flow_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Resolve flats in a tile-based processing pipeline.

    This function is a modified version of the resolve_flats algorithm for use in a tile-based
    processing pipeline. It finds flat edges, labels the flats, and computes gradients away from
    higher and lower terrain within the flats. The computed gradients can be used to determine
    flow directions across the flats.

    Args:
        dem (np.ndarray): Array containing the elevation values for each cell in the tile.
        flow_dirs (np.ndarray): Array containing flow directions for each cell in the tile.
            Cells without a local gradient have a value of FLOW_DIRECTION_UNDEFINED,
            while all other cells have defined flow directions or FLOW_DIRECTION_NODATA.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
            - dist_to_higher: Array containing the gradient values away from higher terrain.
            - dist_to_lower: Array containing the gradient values towards lower terrain.
            - labels: Array indicating which flat each cell is a member of.
            - high_edges: List of high edge cells.
            - low_edges: List of low edge cells.
    """

    # Find flat edges, this will not result in any uncertain edges
    # because we are processing with a 1 cell buffer
    high_edges, low_edges = flat_edges_tile(dem, flow_dirs)

    # remove buffer from dem and flow_dirs since we now have the edges
    dem = dem[1:-1, 1:-1]
    flow_dirs = flow_dirs[1:-1, 1:-1]

    # if the tile is entirely flat, set labels to 1 and return zero gradients
    if (
        len(low_edges) == 0
        and len(high_edges) == 0
        and flow_dirs[0, 0] == FLOW_DIRECTION_UNDEFINED
    ):
        labels = np.ones_like(dem, dtype=np.int64)
        dist_to_higher = np.zeros_like(dem, dtype=np.int64)
        dist_to_lower = np.zeros_like(dem, dtype=np.int64)
        return dist_to_higher, dist_to_lower, labels, high_edges, low_edges

    # Initialize labels array
    labels = np.zeros_like(dem, dtype=np.int64)

    label = 1
    # Label flats from low edges and high edges
    for row, col in low_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1
    for row, col in high_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1

    # Compute gradient away from higher terrain
    dist_to_higher = compute_gradient(labels, flow_dirs, high_edges)
    # Compute gradient away from lower terrain
    dist_to_lower = compute_gradient(labels, flow_dirs, low_edges)

    return dist_to_higher, dist_to_lower, labels, high_edges, low_edges
