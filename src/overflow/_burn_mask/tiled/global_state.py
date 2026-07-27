import numpy as np
from numba import njit  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import Dict  # type: ignore[attr-defined]
from numba.types import DictType, float64, int64

from overflow._burn_mask.core import (
    REGION_STAT_COUNT,
    REGION_STAT_MAX,
    REGION_STAT_MIN,
    REGION_STAT_SUM,
    region_value,
)
from overflow._util.constants import BURN_NO_REGION_LABEL
from overflow._util.perimeter import Float64Perimeter, Int64Perimeter
from overflow._util.raster import Corner, Side
from overflow._util.union_find import UnionFind


@njit
def handle_edge(
    labels_a: np.ndarray,
    masks_a: np.ndarray,
    labels_b: np.ndarray,
    masks_b: np.ndarray,
    union_find: UnionFind,
    connectivity: int,
) -> None:
    """
    Join regions that continue across the shared edge of two adjacent tiles.

    Both label vectors run in the same direction along the seam, so index i in
    tile A sits directly against index i in tile B. With 8-connectivity a cell also
    touches its two diagonal neighbors across the seam, which is why the inner loop
    scans i-1, i and i+1; with 4-connectivity only i is adjacent.

    Two labels are joined when both cells are part of a region and hold the same
    mask value, which is the same rule the within-tile flood fill applies.

    Args:
        labels_a (np.ndarray): Region labels from tile A along the shared edge.
        masks_a (np.ndarray): Mask values from tile A along the shared edge.
        labels_b (np.ndarray): Region labels from tile B along the shared edge.
        masks_b (np.ndarray): Mask values from tile B along the shared edge.
        union_find (UnionFind): The global disjoint set of region labels, modified
            in place.
        connectivity (int): Either 4 or 8.

    Returns:
        None. The function modifies union_find in place.
    """
    for i in range(labels_a.shape[0]):
        label_a = labels_a[i]
        if label_a == BURN_NO_REGION_LABEL:
            continue
        value_a = masks_a[i]
        first = i - 1 if connectivity == 8 else i
        last = i + 1 if connectivity == 8 else i
        for ni in range(first, last + 1):
            if ni < 0 or ni >= labels_b.shape[0]:
                continue
            label_b = labels_b[ni]
            if label_b == BURN_NO_REGION_LABEL:
                continue
            if masks_b[ni] != value_a:
                continue
            union_find.union(label_a, label_b)


@njit
def handle_corner(
    label_a: int64,
    value_a: float64,
    label_b: int64,
    value_b: float64,
    union_find: UnionFind,
) -> None:
    """
    Join regions that continue across the diagonal touch point of two tiles.

    Only reachable with 8-connectivity, and only for the two diagonal pairs meeting
    at the point where four tiles join. Every other diagonal adjacency is already
    covered by handle_edge.

    Args:
        label_a (int64): Region label at tile A's corner.
        value_a (float64): Mask value at tile A's corner.
        label_b (int64): Region label at tile B's corner.
        value_b (float64): Mask value at tile B's corner.
        union_find (UnionFind): The global disjoint set of region labels, modified
            in place.

    Returns:
        None. The function modifies union_find in place.
    """
    if label_a == BURN_NO_REGION_LABEL or label_b == BURN_NO_REGION_LABEL:
        return
    if value_a != value_b:
        return
    union_find.union(label_a, label_b)


@jitclass(
    [
        ("label_perimeters", int64[:, :]),
        ("mask_perimeters", float64[:, :]),
        ("region_min", DictType(int64, float64)),
        ("region_max", DictType(int64, float64)),
        ("region_sum", DictType(int64, float64)),
        ("region_count", DictType(int64, float64)),
    ]
)
class GlobalState:
    """
    Global state for burning per region statistics across a tiled raster.

    Each tile labels its own regions from a private label range, so a region that
    straddles a tile boundary starts out as two or more independent labels with
    independent statistics. This class holds what is needed to put them back
    together: the label and mask values around each tile's perimeter, a disjoint set
    over labels, and the per label statistics.

    The statistics are kept as four parallel dicts rather than one dict of records
    because a numba typed dict has a single value type.

    Attributes:
        union_find (UnionFind): Disjoint set joining labels of the same region.
        label_perimeters (np.ndarray): Per tile region labels around the perimeter.
        mask_perimeters (np.ndarray): Per tile mask values around the perimeter,
            held as float64 so one class covers every mask raster data type.
        region_min (DictType(int64, float64)): Per label minimum DEM value.
        region_max (DictType(int64, float64)): Per label maximum DEM value.
        region_sum (DictType(int64, float64)): Per label sum of DEM values.
        region_count (DictType(int64, float64)): Per label count of valid DEM cells.
        num_rows (int64): Number of tile rows.
        num_cols (int64): Number of tile columns.
        chunk_size (int64): Size of each tile in cells.
        connectivity (int64): Either 4 or 8.
    """

    union_find: UnionFind
    label_perimeters: np.ndarray
    mask_perimeters: np.ndarray
    region_min: DictType(int64, float64)  # type: ignore[valid-type]
    region_max: DictType(int64, float64)  # type: ignore[valid-type]
    region_sum: DictType(int64, float64)  # type: ignore[valid-type]
    region_count: DictType(int64, float64)  # type: ignore[valid-type]
    num_rows: int64
    num_cols: int64
    chunk_size: int64
    connectivity: int64

    def __init__(
        self, num_rows: int64, num_cols: int64, chunk_size: int64, connectivity: int64
    ):
        self.union_find = UnionFind()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.chunk_size = chunk_size
        self.connectivity = connectivity
        self.region_min = Dict.empty(key_type=int64, value_type=float64)
        self.region_max = Dict.empty(key_type=int64, value_type=float64)
        self.region_sum = Dict.empty(key_type=int64, value_type=float64)
        self.region_count = Dict.empty(key_type=int64, value_type=float64)
        tile_count = num_rows * num_cols
        perimeter_cell_count = 4 * chunk_size - 4
        self.label_perimeters = np.full(
            (tile_count, perimeter_cell_count), BURN_NO_REGION_LABEL, dtype=np.int64
        )
        self.mask_perimeters = np.zeros(
            (tile_count, perimeter_cell_count), dtype=np.float64
        )

    def row_col_to_tile_index(self, row: int64, col: int64) -> int64:
        """Return the flat tile index of the tile at the given tile row and column."""
        return row * self.num_cols + col

    def _get_label_perimeter(self, tile_index: int64) -> Int64Perimeter:
        return Int64Perimeter(
            self.label_perimeters[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def _get_mask_perimeter(self, tile_index: int64) -> Float64Perimeter:
        return Float64Perimeter(
            self.mask_perimeters[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def merge_tile_stats(self, label_offset: int64, stats: np.ndarray) -> None:
        """Record one tile's per region statistics.

        Labels are unique to the tile that created them, so this is a plain insert
        with no merging. Callers must serialize this, since numba typed dicts are
        not safe to insert into concurrently.

        Args:
            label_offset (int64): The first label used by the tile.
            stats (np.ndarray): The tile's (region_count, 4) statistics array.
        """
        for index in range(stats.shape[0]):
            label = int64(label_offset + index)
            self.region_min[label] = stats[index, REGION_STAT_MIN]
            self.region_max[label] = stats[index, REGION_STAT_MAX]
            self.region_sum[label] = stats[index, REGION_STAT_SUM]
            self.region_count[label] = stats[index, REGION_STAT_COUNT]

    def connect_tile_edges_and_corners(self) -> None:
        """Join every pair of labels that meet across a tile boundary.

        Walks each tile once and joins it to its eastern and southern neighbors,
        plus the two diagonal pairs meeting at its southeastern corner. Every
        adjacency in the tile grid is covered exactly once, and the bounds guards
        keep it correct for degenerate grids that are a single tile wide or tall.
        """
        for row_index in range(self.num_rows):
            for col_index in range(self.num_cols):
                # + - - + - - +
                # |  A  |  B  |
                # + - - * - - +
                # |  C  |  D  |
                # + - - + - - +
                tile_index_a = self.row_col_to_tile_index(row_index, col_index)
                has_east = col_index + 1 < self.num_cols
                has_south = row_index + 1 < self.num_rows
                if has_east:
                    tile_index_b = self.row_col_to_tile_index(row_index, col_index + 1)
                    self._combine_edge(
                        tile_index_a, Side.RIGHT, tile_index_b, Side.LEFT
                    )
                if has_south:
                    tile_index_c = self.row_col_to_tile_index(row_index + 1, col_index)
                    self._combine_edge(
                        tile_index_a, Side.BOTTOM, tile_index_c, Side.TOP
                    )
                if has_east and has_south and self.connectivity == 8:
                    tile_index_b = self.row_col_to_tile_index(row_index, col_index + 1)
                    tile_index_c = self.row_col_to_tile_index(row_index + 1, col_index)
                    tile_index_d = self.row_col_to_tile_index(
                        row_index + 1, col_index + 1
                    )
                    self._combine_corner(
                        tile_index_a,
                        Corner.BOTTOM_RIGHT,
                        tile_index_d,
                        Corner.TOP_LEFT,
                    )
                    self._combine_corner(
                        tile_index_b,
                        Corner.BOTTOM_LEFT,
                        tile_index_c,
                        Corner.TOP_RIGHT,
                    )

    def _combine_edge(
        self, tile_index_a: int64, side_a: Side, tile_index_b: int64, side_b: Side
    ) -> None:
        """Join regions across the shared edge of two adjacent tiles."""
        handle_edge(
            self._get_label_perimeter(tile_index_a).get_side(side_a),
            self._get_mask_perimeter(tile_index_a).get_side(side_a),
            self._get_label_perimeter(tile_index_b).get_side(side_b),
            self._get_mask_perimeter(tile_index_b).get_side(side_b),
            self.union_find,
            self.connectivity,
        )

    def _combine_corner(
        self,
        tile_index_a: int64,
        corner_a: Corner,
        tile_index_b: int64,
        corner_b: Corner,
    ) -> None:
        """Join regions across the diagonal touch point of two tiles."""
        handle_corner(
            self._get_label_perimeter(tile_index_a).get_corner(corner_a),
            self._get_mask_perimeter(tile_index_a).get_corner(corner_a),
            self._get_label_perimeter(tile_index_b).get_corner(corner_b),
            self._get_mask_perimeter(tile_index_b).get_corner(corner_b),
            self.union_find,
        )

    def solve_regions(self, statistic: int64, burn_offset: float64):
        """Merge per tile statistics into whole regions and reduce them to values.

        Every label is resolved to its representative, the four mergeable
        statistics are combined per representative, and the result is expanded back
        into a flat lookup so the apply pass is a single dict read per cell.

        Args:
            statistic (int64): One of the BURN_STAT_* constants.
            burn_offset (float64): Subtracted from each computed statistic.

        Returns:
            Dict: A numba typed dict mapping every label to its region's burn value.
                Regions with no valid DEM cells map to NaN, meaning leave alone.
        """
        root_min = Dict.empty(key_type=int64, value_type=float64)
        root_max = Dict.empty(key_type=int64, value_type=float64)
        root_sum = Dict.empty(key_type=int64, value_type=float64)
        root_count = Dict.empty(key_type=int64, value_type=float64)
        for label in self.region_min:
            root = self.union_find.find(label)
            if root in root_min:
                if self.region_min[label] < root_min[root]:
                    root_min[root] = self.region_min[label]
                if self.region_max[label] > root_max[root]:
                    root_max[root] = self.region_max[label]
                root_sum[root] += self.region_sum[label]
                root_count[root] += self.region_count[label]
            else:
                root_min[root] = self.region_min[label]
                root_max[root] = self.region_max[label]
                root_sum[root] = self.region_sum[label]
                root_count[root] = self.region_count[label]
        burn_lookup = Dict.empty(key_type=int64, value_type=float64)
        for label in self.region_min:
            root = self.union_find.find(label)
            burn_lookup[label] = region_value(
                root_min[root],
                root_max[root],
                root_sum[root],
                root_count[root],
                statistic,
                burn_offset,
            )
        return burn_lookup
