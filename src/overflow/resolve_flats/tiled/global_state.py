import concurrent.futures
import math
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import List  # type: ignore[attr-defined]
from osgeo import gdal

from overflow.resolve_flats.tiled.graphs import GlobalGraph
from overflow.util.numba_types import Int64Pair, Int64PairListList
from overflow.util.perimeter import (
    Float32Perimeter,
    Float32PerimeterList,
    get_tile_perimeter,
)
from overflow.util.raster import RasterChunk, raster_chunker

from .flat_distance import compute_min_dist_flood
from .graphs import LocalGraph
from .resolve_flats import compute_labels_and_gradients


@jitclass
class GlobalState:
    """
    Represents the global state of the DEM processing.

    This class holds the global graph, elevation perimeters, and interior edge coordinates
    for each tile in the DEM. It provides methods to initialize the state, retrieve tile
    indices, and complete the global graph by joining adjacent tiles.

    Attributes:
        graph (GlobalGraph): The global graph representing the connectivity of tiles.
        elevations (Float32PerimeterList): List of elevation perimeters for each tile.
        interior_low_edge_cells (Int64PairListList): List of interior low edge cell coordinates for each tile.
        interior_high_edge_cells (Int64PairListList): List of interior high edge cell coordinates for each tile.
    """

    graph: GlobalGraph
    elevations: Float32PerimeterList
    interior_low_edges: Int64PairListList
    interior_high_edges: Int64PairListList

    def __init__(self, tile_rows: int, tile_cols: int, perimeter_cells_per_tile: int):
        self.graph = GlobalGraph(tile_rows, tile_cols, perimeter_cells_per_tile)
        tile_count = tile_rows * tile_cols
        self.elevations = List(
            [Float32Perimeter(np.empty(0, dtype=np.float32), 0, 0, 0)] * tile_count
        )
        self.interior_low_edges = List([List.empty_list(Int64Pair)] * tile_count)
        self.interior_high_edges = List([List.empty_list(Int64Pair)] * tile_count)

    def _get_tile_index(self, row: int, col: int) -> int:
        return row * self.graph.tile_cols + col

    def complete_graph(self):
        """
        Complete the global graph by joining adjacent tiles.

        This method iterates over each pair of adjacent tiles in the DEM and joins their
        edges and corners using the `join_adjacent_tiles` method of the global graph.

        The adjacent tiles are represented as follows:
        + - - + - - +
        |  A  |  B  |
        + - - * - - +
        |  C  |  D  |
        + - - + - - +
        """
        for tile_row in range(self.graph.tile_rows - 1):
            for tile_col in range(self.graph.tile_cols - 1):
                elevations_a = self.elevations[self._get_tile_index(tile_row, tile_col)]
                elevations_b = self.elevations[
                    self._get_tile_index(tile_row, tile_col + 1)
                ]
                elevations_c = self.elevations[
                    self._get_tile_index(tile_row + 1, tile_col)
                ]
                elevations_d = self.elevations[
                    self._get_tile_index(tile_row + 1, tile_col + 1)
                ]
                self.graph.join_adjacent_tiles(
                    elevations_a, elevations_b, elevations_c, elevations_d
                )


@njit(nogil=True)
def create_local_graph(
    dem: np.ndarray, fdr: np.ndarray, tile_index: int
) -> tuple[LocalGraph, list, list, np.ndarray]:
    """
    Create a local graph for a tile in the DEM.

    This function computes the labels, gradients, and distances between edge cells belonging
    to the same flat in a single tile of the DEM. It then constructs a LocalGraph object
    representing the local connectivity of the tile and returns the local graph along with
    the high edges, low edges, and labels.

    Args:
        dem (np.ndarray): The elevation data for the tile.
        fdr (np.ndarray): The flow direction data for the tile.
        tile_index (int): The index of the tile in the global DEM.

    Returns:
        tuple: A tuple containing the following elements:
            - local_graph (LocalGraph): The local graph representing the connectivity of the tile.
            - high_edges (list): The list of high edge cells in the tile.
            - low_edges (list): The list of low edge cells in the tile.
            - labels (np.ndarray): The labels assigned to each cell in the tile.
    """
    to_higher, to_lower, labels, high_edges, low_edges = compute_labels_and_gradients(
        dem, fdr
    )
    # compute distances between edge cells belonging to the same flat in the same tile
    # this will be used to construct the local graph
    min_dists = compute_min_dist_flood(labels)
    # min_dists = compute_min_dist_bfs(labels)
    to_higher_perimeter = get_tile_perimeter(to_higher)
    to_lower_perimeter = get_tile_perimeter(to_lower)
    labels_perimeter = get_tile_perimeter(labels)
    local_graph = LocalGraph(
        labels_perimeter,
        to_higher_perimeter,
        to_lower_perimeter,
        min_dists,
        tile_index,
    )
    return local_graph, high_edges, low_edges, labels


def create_global_state(
    dem_band: gdal.Band,
    fdr_band: gdal.Band,
    labels_band: gdal.Band,
    chunk_size: int,
    progress_callback=None,
) -> GlobalState:
    """
    Create a GlobalState object representing the global state of the DEM processing.

    This function is part of the larger "fix_flats_tiled" process to fix flats in a DEM
    using a tiled approach.
    It initializes the GlobalState object by processing the DEM in chunks, computing labels,
    gradients, and constructing the local graphs for each chunk. The local graphs are then
    added to the global graph, and the interior low and high edges are stored in the
    GlobalState object. Finally, the global graph is completed by joining cells beloinging to
    the same flats in separate tiles, and the labels are written to disk for use later.

    Args:
        dem_band (gdal.Band): The GDAL Band object representing the DEM.
        fdr_band (gdal.Band): The GDAL Band object representing the flow direction raster.
        labels_band (gdal.Band): The GDAL Band object to store the computed labels.
        chunk_size (int): The size of each chunk (tile) to process the DEM.

    Returns:
        GlobalState: The constructed GlobalState object representing the global state
            of the DEM processing.

    Note:
        - The function is designed to be executed in a parallel or multi-threaded environment,
          where the computation of labels, gradients, and local graphs for each chunk can be
          performed independently.
    """
    tile_rows = math.ceil(dem_band.YSize / chunk_size)
    tile_cols = math.ceil(dem_band.XSize / chunk_size)
    perimeter_cells_per_tile = 4 * (chunk_size - 2) + 4
    global_state = GlobalState(tile_rows, tile_cols, perimeter_cells_per_tile)
    tile_index = 0
    tile_index_map: dict[int, tuple[int, int]] = {}

    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    # Track progress as tiles are processed
    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = tile_rows * tile_cols

    def handle_result(future):
        with lock:
            local_graph, high_edges, low_edges, labels = future.result()
            global_state.graph.add(local_graph)
            global_state.interior_low_edges[local_graph.tile_index] = low_edges
            global_state.interior_high_edges[local_graph.tile_index] = high_edges
            row, col = tile_index_map[local_graph.tile_index]
            labels_tile = RasterChunk(row, col, chunk_size, 0)
            labels_tile.from_numpy(labels)
            labels_tile.write(labels_band)
            task_queue.get()
            # Report progress as tiles complete processing
            chunk_counter[0] += 1
            if progress_callback is not None:
                progress_callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dem_tile in raster_chunker(dem_band, chunk_size, 1):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(tile_index)
            fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
            fdr_tile.read(fdr_band)
            assert dem_tile.data is not None
            dem_unbuffered = dem_tile.data[1:-1, 1:-1]
            perimeter_elevations = get_tile_perimeter(dem_unbuffered)
            global_state.elevations[tile_index] = Float32Perimeter(
                perimeter_elevations, chunk_size, chunk_size, tile_index
            )

            future = executor.submit(
                create_local_graph, dem_tile.data, fdr_tile.data, tile_index
            )
            tile_index_map[tile_index] = (dem_tile.row, dem_tile.col)
            future.add_done_callback(handle_result)

            tile_index += 1

    while not task_queue.empty():
        time.sleep(0.1)

    global_state.complete_graph()
    labels_band.FlushCache()
    return global_state
