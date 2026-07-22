import math

import numpy as np
from numba import njit, prange  # type: ignore[attr-defined]
from osgeo import gdal

from overflow._resolve_flats.tiled.flat_mask import MAX_FLAT_HEIGHT
from overflow._util.constants import (
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTIONS,
    NEIGHBOR_OFFSETS,
)
from overflow._util.perimeter import Int64Perimeter
from overflow._util.raster import RasterChunk, raster_chunker

# Marks seam-mask cells with no globally-solved value (no neighboring tile, or
# the cell is not a flat perimeter cell). Comparisons fall back to raw masks.
SEAM_MASK_UNKNOWN = np.int64(np.iinfo(np.int64).min)


@njit
def _canonical_mask(dist_to_high_edge: int, dist_to_low_edge: int) -> int:
    """Reconstruct a tile-perimeter cell's flat mask from its globally solved
    distances, using the same four cases as towards_lower_tile. Solved
    distances are 0 when the cell has no path to that edge type.

    The per-tile masks produced by away_from_higher_tile/towards_lower_tile
    equal this value only when the tile's seeding matched the global solution;
    this canonical form is what makes values comparable across tile seams.
    """
    if dist_to_high_edge > 0 and dist_to_low_edge > 0:
        return MAX_FLAT_HEIGHT - dist_to_high_edge + 2 * dist_to_low_edge
    if dist_to_high_edge > 0:
        return -dist_to_high_edge
    if dist_to_low_edge > 0:
        return 2 * dist_to_low_edge
    return 0


@njit
def build_seam_canonical_mask(
    dist_to_high_edge_tiles: np.ndarray,
    dist_to_low_edge_tiles: np.ndarray,
    tile_row: int,
    tile_col: int,
    tile_rows: int,
    tile_cols: int,
    chunk_size: int,
) -> np.ndarray:
    """Build a (chunk_size+2)^2 array of canonical mask values aligned with a
    buffer-1 tile read: the tile's own perimeter ring plus the halo ring taken
    from the 8 neighboring tiles' perimeters. Cells without a solved value
    (including the tile interior) hold SEAM_MASK_UNKNOWN.
    """
    size = chunk_size + 2
    seam = np.full((size, size), SEAM_MASK_UNKNOWN, dtype=np.int64)
    n = chunk_size

    # the tile's own outer ring
    tile_index = tile_row * tile_cols + tile_col
    perimeter = Int64Perimeter(dist_to_low_edge_tiles[tile_index], n, n, tile_index)
    for i in range(perimeter.size()):
        row, col = perimeter.get_row_col(i)
        seam[row + 1, col + 1] = _canonical_mask(
            dist_to_high_edge_tiles[tile_index][i],
            dist_to_low_edge_tiles[tile_index][i],
        )

    # west neighbor's east column
    if tile_col > 0:
        idx = tile_index - 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        for i in range(n):
            j = p.get_index(i, n - 1)
            seam[i + 1, 0] = _canonical_mask(
                dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
            )
    # east neighbor's west column
    if tile_col < tile_cols - 1:
        idx = tile_index + 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        for i in range(n):
            j = p.get_index(i, 0)
            seam[i + 1, size - 1] = _canonical_mask(
                dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
            )
    # north neighbor's bottom row
    if tile_row > 0:
        idx = tile_index - tile_cols
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        for i in range(n):
            j = p.get_index(n - 1, i)
            seam[0, i + 1] = _canonical_mask(
                dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
            )
    # south neighbor's top row
    if tile_row < tile_rows - 1:
        idx = tile_index + tile_cols
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        for i in range(n):
            j = p.get_index(0, i)
            seam[size - 1, i + 1] = _canonical_mask(
                dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
            )
    # corner neighbors
    if tile_row > 0 and tile_col > 0:
        idx = tile_index - tile_cols - 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        j = p.get_index(n - 1, n - 1)
        seam[0, 0] = _canonical_mask(
            dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
        )
    if tile_row > 0 and tile_col < tile_cols - 1:
        idx = tile_index - tile_cols + 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        j = p.get_index(n - 1, 0)
        seam[0, size - 1] = _canonical_mask(
            dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
        )
    if tile_row < tile_rows - 1 and tile_col > 0:
        idx = tile_index + tile_cols - 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        j = p.get_index(0, n - 1)
        seam[size - 1, 0] = _canonical_mask(
            dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
        )
    if tile_row < tile_rows - 1 and tile_col < tile_cols - 1:
        idx = tile_index + tile_cols + 1
        p = Int64Perimeter(dist_to_low_edge_tiles[idx], n, n, idx)
        j = p.get_index(0, 0)
        seam[size - 1, size - 1] = _canonical_mask(
            dist_to_high_edge_tiles[idx][j], dist_to_low_edge_tiles[idx][j]
        )
    return seam


@njit(parallel=True)
def d8_masked_flow_dirs_seam(
    dem: np.ndarray, flat_mask: np.ndarray, fdr: np.ndarray, seam_mask: np.ndarray
) -> None:
    """Tile-aware variant of d8_masked_flow_dirs (Algorithm 7) operating on a
    buffer-1 tile read. Identical to the core version except:

    - only interior cells are assigned (halo assignments were discarded on
      write anyway);
    - when the candidate neighbor is a halo cell, the slope is computed from
      globally consistent canonical mask values for both cells instead of the
      raw per-tile masks, which are on non-comparable scales across seams;
    - a halo candidate with exactly zero slope is only eligible when it
      precedes the current cell in row-major order, so two cells on opposite
      sides of a seam can never both select each other.
    """
    rows, cols = fdr.shape
    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):
            if fdr[row, col] != FLOW_DIRECTION_UNDEFINED:
                continue
            nmin = FLOW_DIRECTION_UNDEFINED
            min_slope = np.inf
            for i, (d_row, d_col) in enumerate(NEIGHBOR_OFFSETS):
                neighbor_row = row + d_row
                neighbor_col = col + d_col
                # if the neighbor is nodata, drain to nodata
                if fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_NODATA:
                    nmin = i
                    min_slope = -np.inf
                    break
                # if the fdr is not part of the same flat, skip
                if dem[neighbor_row, neighbor_col] != dem[row, col]:
                    continue
                is_halo = (
                    neighbor_row == 0
                    or neighbor_row == rows - 1
                    or neighbor_col == 0
                    or neighbor_col == cols - 1
                )
                if (
                    is_halo
                    and seam_mask[neighbor_row, neighbor_col] != SEAM_MASK_UNKNOWN
                    and seam_mask[row, col] != SEAM_MASK_UNKNOWN
                ):
                    dz = float(
                        seam_mask[neighbor_row, neighbor_col] - seam_mask[row, col]
                    )
                else:
                    dz = (
                        float(flat_mask[neighbor_row, neighbor_col])
                        - flat_mask[row, col]
                    )
                slope = dz / (math.sqrt(2) if d_row != 0 and d_col != 0 else 1)
                if (
                    is_halo
                    and slope == 0
                    and not (
                        neighbor_row < row
                        or (neighbor_row == row and neighbor_col < col)
                    )
                ):
                    # deterministic cross-seam tie-break: only the side for
                    # which the neighbor precedes it in row-major order may
                    # take a zero-slope step across the seam
                    continue
                # update minimum slope
                if slope < min_slope:
                    min_slope = slope
                    nmin = FLOW_DIRECTIONS[i]
            fdr[row, col] = nmin


def update_fdr(
    dem_band: gdal.Band,
    fdr_band: gdal.Band,
    fixed_fdr_band: gdal.Band,
    flat_mask_band: gdal.Band,
    chunk_size: int,
    dist_to_high_edge_tiles: np.ndarray,
    dist_to_low_edge_tiles: np.ndarray,
    progress_callback=None,
):
    # update fdr using d8_masked_flow_dirs_seam
    tile_rows = math.ceil(fdr_band.YSize / chunk_size)
    tile_cols = math.ceil(fdr_band.XSize / chunk_size)
    for fdr_tile in raster_chunker(
        fdr_band, chunk_size, 1, progress_callback=progress_callback
    ):
        dem_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        dem_tile.read(dem_band)
        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        flat_mask_tile.read(flat_mask_band)
        seam_mask = build_seam_canonical_mask(
            dist_to_high_edge_tiles,
            dist_to_low_edge_tiles,
            fdr_tile.row,
            fdr_tile.col,
            tile_rows,
            tile_cols,
            chunk_size,
        )
        d8_masked_flow_dirs_seam(
            dem_tile.data, flat_mask_tile.data, fdr_tile.data, seam_mask
        )
        fdr_tile.write(fixed_fdr_band)
    fixed_fdr_band.FlushCache()
