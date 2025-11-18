import concurrent.futures
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from numba.typed import List  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.fix_flats.tiled.global_state import GlobalState
from overflow.util.constants import FLOW_DIRECTION_UNDEFINED
from overflow.util.numba_types import Int64PairList
from overflow.util.perimeter import Int64Perimeter
from overflow.util.queue import Int64PairQueue as Queue
from overflow.util.raster import RasterChunk, neighbor_generator, raster_chunker

# MAX_FLAT_HEIGHT can be any arbitrary large number
# as long as is is greater than the maximum distance
# to any high edge cell in the flat
MAX_FLAT_HEIGHT = (2**31 - 1) // 2


@njit
def get_edge_exterior_distances(
    dist_to_edge_perimeter: Int64Perimeter,
) -> tuple[list, int, dict]:
    """
    This function formats the distances to edge cells in the global graph for an individial tile.

    Args:
        dist_to_edge_perimeter (Int64Perimeter): The distances from each perimeter cell to the nearest
            edge cell in the global graph.

    Returns:
        List: A list of lists containing the exterior edge cells for each loop.
        int: The maximum loop value. This is the maximum distance to an edge cell.
        Dict: A mapping of loop values to their index in the exterior edge list.
    """
    exterior_edges = List()
    max_loop = 0
    loop_index_map = {}
    index = 0
    for i, dist in enumerate(dist_to_edge_perimeter.data):
        if dist <= 1:
            continue
        row, col = dist_to_edge_perimeter.get_row_col(i)
        if dist not in loop_index_map:
            max_loop = max(max_loop, dist)
            loop_index_map[dist] = index
            index += 1
            new_edge = List()
            new_edge.append((row, col))
            exterior_edges.append(new_edge)
        else:
            exterior_edges[loop_index_map[dist]].append((row, col))
    return exterior_edges, max_loop, loop_index_map


@njit
def away_from_higher_tile(
    interior_high_edges: Int64PairList,
    dist_to_high_edge_perimeter: Int64Perimeter,
    labels: np.ndarray,
    fdr: np.ndarray,
    flat_mask: np.ndarray,
):
    """
    Tiled version of Algorithm 5 AwayFromHigher.

    This procedure builds a gradient away from higher terrain within a tile, considering the
    distances to the high edge cells from cells in the global graph.

    Upon entry:
    (1) interior_high_edges contains, in no particular order, all the high edge cells within the
        tile which are part of drainable flats.
    (2) dist_to_high_edge_perimeter contains the distances from each perimeter cell to the nearest
        high edge cell in the global graph.
    (3) labels contains the labels of each flat within the tile, same shape as the tile. 0 means
        not part of a flat.
    (4) fdr contains the flow direction raster for the tile.
    (5) flat_mask is initialized to 0 for all cells within the tile.

    At exit:
    (1) flat_mask contains the number of increments to be applied to each cell within the tile to
        form a gradient away from higher terrain; cells not in a flat have a value of 0.

    Args:
        interior_high_edges (list): The high edge cells within the tile. In no particular order.
        dist_to_high_edge_perimeter (np.ndarray): The distances from each perimeter cell to the
            nearest high edge cell in the global graph.
        labels (np.ndarray): The labels of each flat within the tile, same shape as the tile.
            0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster for the tile.
        flat_mask (np.ndarray): The flat mask for the tile, same shape as the tile. 0 means not
            part of a flat.

    Returns:
        None
    """
    exterior_high_edges, max_high_loop, loop_index_map = get_edge_exterior_distances(
        dist_to_high_edge_perimeter
    )
    marker = (-1, -1)
    interior_high_edges.append(marker)
    interior_high_edges = Queue(interior_high_edges)
    loops = 1
    while len(interior_high_edges) > 1 or loops < max_high_loop:
        row, col = interior_high_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            if loops in loop_index_map:
                for row, col in exterior_high_edges[loop_index_map[loops]]:
                    interior_high_edges.push((row, col))
            interior_high_edges.push(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        flat_mask[row, col] = loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, labels.shape[0], labels.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                interior_high_edges.push((neighbor_row, neighbor_col))


@njit
def towards_lower_tile(
    interior_low_edges: Int64PairList,
    dist_to_low_edge_perimeter: Int64Perimeter,
    labels: np.ndarray,
    fdr: np.ndarray,
    flat_mask: np.ndarray,
):
    """
    Tiled version of Algorithm 6 TowardsLower.

    This procedure builds a gradient towards lower terrain within a tile, considering the
    distances to the low edge cells from cells in the global graph.

    Upon entry:
    (1) interior_low_edges contains, in no particular order, all the low edge cells within the
        tile which are part of drainable flats.
    (2) dist_to_low_edge_perimeter contains the distances from each perimeter cell to the nearest
        low edge cell in the global graph.
    (3) labels contains the labels of each flat within the tile, same shape as the tile. 0 means
        not part of a flat.
    (4) fdr contains the flow direction raster for the tile.
    (5) flat_mask is initialized to 0 for all cells within the tile.

    At exit:
    (1) flat_mask contains the number of increments to be applied to each cell within the tile to
        form a gradient towards lower terrain; cells not in a flat have a value of 0.

    Args:
        interior_low_edges (list): The low edge cells within the tile. In no particular order.
        dist_to_low_edge_perimeter (np.ndarray): The distances from each perimeter cell to the
            nearest low edge cell in the global graph.
        labels (np.ndarray): The labels of each flat within the tile, same shape as the tile.
            0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster for the tile.
        flat_mask (np.ndarray): The flat mask for the tile, same shape as the tile. 0 means not
            part of a flat.

    Returns:
        None
    """
    exterior_low_edges, max_low_loop, loop_index_map = get_edge_exterior_distances(
        dist_to_low_edge_perimeter
    )
    marker = (-1, -1)
    interior_low_edges.append(marker)
    interior_low_edges = Queue(interior_low_edges)
    loops = 1
    flat_mask *= -1

    while len(interior_low_edges) > 1 or loops < max_low_loop:
        row, col = interior_low_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            if loops in loop_index_map:
                for row, col in exterior_low_edges[loop_index_map[loops]]:
                    interior_low_edges.push((row, col))
            interior_low_edges.push(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        if flat_mask[row, col] < 0:
            flat_mask[row, col] += MAX_FLAT_HEIGHT + 2 * loops
        else:
            flat_mask[row, col] = 2 * loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, labels.shape[0], labels.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                interior_low_edges.push((neighbor_row, neighbor_col))


@njit(nogil=True)
def create_flat_mask_tile(
    interior_high_edges: Int64PairList,
    interior_low_edges: Int64PairList,
    dist_to_low_edge: np.ndarray,
    dist_to_high_edge: np.ndarray,
    fdr: np.ndarray,
    labels: np.ndarray,
    tile_index: int,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    """
    Create a flat mask for a given tile.

    Args:
        interior_high_edges (Int64PairList): Interior high edges list.
        interior_low_edges (Int64PairList): Interior low edges list.
        dist_to_low_edge (np.ndarray): Distance to low edge array.
        dist_to_high_edge (np.ndarray): Distance to high edge array.
        fdr (np.ndarray): Flow direction raster.
        labels (np.ndarray): Labels raster.
        tile_index (int): Index of the current tile.
        chunk_size (int): Size of the tile chunk.

    Returns:
        np.ndarray: Flat mask array.
    """
    flat_mask = np.zeros_like(fdr, dtype=np.int64)
    dist_to_high_edge_perimeter = Int64Perimeter(
        dist_to_high_edge,
        chunk_size,
        chunk_size,
        tile_index,
    )
    dist_to_low_edge_perimeter = Int64Perimeter(
        dist_to_low_edge,
        chunk_size,
        chunk_size,
        tile_index,
    )
    away_from_higher_tile(
        interior_high_edges,
        dist_to_high_edge_perimeter,
        labels,
        fdr,
        flat_mask,
    )
    towards_lower_tile(
        interior_low_edges,
        dist_to_low_edge_perimeter,
        labels,
        fdr,
        flat_mask,
    )
    return flat_mask, tile_index


def create_flat_mask(
    chunk_size: int,
    flat_mask_band: gdal.Band,
    labels_band: gdal.Band,
    fdr_band: gdal.Band,
    global_state: GlobalState,
    dist_to_high_edge_tiles: np.ndarray,
    dist_to_low_edge_tiles: np.ndarray,
):
    """
    Create a flat mask for the entire raster.

    Args:
        chunk_size (int): Size of the tile chunk.
        flat_mask_band (gdal.Band): Band object for writing flat mask.
        labels_band (gdal.Band): Band object for labels.
        fdr_band (gdal.Band): Band object for flow direction raster.
        global_state (GlobalState): Global state object.
        dist_to_high_edge_tiles (np.ndarray): Distance to high edge tiles.
        dist_to_low_edge_tiles (np.ndarray): Distance to low edge tiles.

    Returns:
        None
    """
    tile_index = 0
    tile_index_map: dict[int, tuple[int, int]] = {}
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    def handle_result(future):
        with lock:
            flat_mask, tile_index = future.result()
            row, col = tile_index_map[tile_index]
            flat_mask_tile = RasterChunk(row, col, chunk_size, 0)
            flat_mask_tile.from_numpy(flat_mask)
            flat_mask_tile.write(flat_mask_band)
            task_queue.get()

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        for fdr_tile in raster_chunker(fdr_band, chunk_size, 0):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(tile_index)
            labels_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 0)
            labels_tile.read(labels_band)
            interior_high_edges = global_state.interior_high_edges[tile_index]
            interior_low_edges = global_state.interior_low_edges[tile_index]
            dist_to_high_edge = dist_to_high_edge_tiles[tile_index]
            dist_to_low_edge = dist_to_low_edge_tiles[tile_index]

            future = executor.submit(
                create_flat_mask_tile,
                interior_high_edges,
                interior_low_edges,
                dist_to_low_edge,
                dist_to_high_edge,
                fdr_tile.data,
                labels_tile.data,
                tile_index,
                chunk_size,
            )
            tile_index_map[tile_index] = (fdr_tile.row, fdr_tile.col)
            future.add_done_callback(handle_result)
            tile_index += 1

    while not task_queue.empty():
        time.sleep(0.1)

    flat_mask_band.FlushCache()
