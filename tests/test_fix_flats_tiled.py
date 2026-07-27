import contextlib

import numpy as np
import pytest
from osgeo import gdal

from overflow._flow_direction.flow_direction import flow_direction_for_tile
from overflow._resolve_flats.core.resolve_flats import d8_masked_flow_dirs, fix_flats
from overflow._resolve_flats.tiled.flat_mask import MAX_FLAT_HEIGHT
from overflow._resolve_flats.tiled.resolve_flats_tiled import _resolve_flats_tiled
from overflow._resolve_flats.tiled.update_fdr import (
    SEAM_MASK_UNKNOWN,
    _canonical_mask,
    build_seam_canonical_mask,
    d8_masked_flow_dirs_seam,
)
from overflow._util.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_WEST,
    NEIGHBOR_OFFSETS,
)
from overflow._util.perimeter import Int64Perimeter

DEM_NODATA = -9999.0


def make_initial_fdr(dem: np.ndarray, nodata: float = DEM_NODATA) -> np.ndarray:
    """Compute the initial D8 flow directions for a DEM the same way the
    pipeline does: cells at the raster edge drain outward, cells without a
    downhill neighbor are FLOW_DIRECTION_UNDEFINED."""
    padded = np.full((dem.shape[0] + 2, dem.shape[1] + 2), nodata, dtype=dem.dtype)
    padded[1:-1, 1:-1] = dem
    return flow_direction_for_tile(padded, nodata)[1:-1, 1:-1]


def priority_flood_fill(dem: np.ndarray) -> np.ndarray:
    """Fill closed depressions with a simple priority flood so the DEM drains
    to its edges, as the pipeline's fill step guarantees before flow
    directions are computed. Filling creates large flats, which is exactly
    what the flat resolution tests need."""
    import heapq

    rows, cols = dem.shape
    filled = dem.copy()
    visited = np.zeros(dem.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in (0, cols - 1):
            heapq.heappush(heap, (filled[r, c], r, c))
            visited[r, c] = True
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            heapq.heappush(heap, (filled[r, c], r, c))
            visited[r, c] = True
    while heap:
        elev, r, c = heapq.heappop(heap)
        for d_row, d_col in NEIGHBOR_OFFSETS:
            nr, nc = r + d_row, c + d_col
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                filled[nr, nc] = max(filled[nr, nc], elev)
                heapq.heappush(heap, (filled[nr, nc], nr, nc))
    return filled


def find_reciprocal_pairs(fdr: np.ndarray) -> list:
    """Find pairs of adjacent cells whose flow directions point at each other
    (a two-cell cycle). Returns a list of ((r1, c1), (r2, c2)) tuples."""
    pairs = []
    rows, cols = fdr.shape
    # checking E, NE, N, NW covers every adjacent pair exactly once;
    # the opposite direction code is d + 4
    for d in range(4):
        d_row, d_col = NEIGHBOR_OFFSETS[d]
        r0, r1 = max(0, -d_row), rows - max(0, d_row)
        c0, c1 = max(0, -d_col), cols - max(0, d_col)
        cell = fdr[r0:r1, c0:c1]
        neighbor = fdr[r0 + d_row : r1 + d_row, c0 + d_col : c1 + d_col]
        for r, c in zip(*np.nonzero((cell == d) & (neighbor == d + 4))):
            pairs.append(
                ((int(r) + r0, int(c) + c0), (int(r) + r0 + d_row, int(c) + c0 + d_col))
            )
    return pairs


def assert_valid_drainage(fdr: np.ndarray) -> None:
    """Assert every cell with a defined direction reaches a terminal (nodata,
    undefined, or off-raster) without cycling."""
    rows, cols = fdr.shape
    max_steps = rows * cols
    for row, col in np.ndindex(fdr.shape):
        r, c = row, col
        for _ in range(max_steps + 1):
            d = fdr[r, c]
            if d >= 8:
                break
            r, c = r + NEIGHBOR_OFFSETS[d][0], c + NEIGHBOR_OFFSETS[d][1]
            if not (0 <= r < rows and 0 <= c < cols):
                break
        else:
            raise AssertionError(f"flow path from ({row}, {col}) cycles")


@pytest.fixture(name="dem_zhou_2022")
def fixture_dem_zhou_2022():
    """DEM from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [5, 4, 4, 5, 9, 7, 2, 4],
            [9, 3, 3, 3, 3, 3, 3, 7],
            [7, 3, 3, 3, 3, 3, 3, 5],
            [8, 3, 3, 3, 3, 3, 3, 6],
            [9, 3, 3, 3, 3, 3, 3, 2],
            [6, 3, 3, 3, 3, 3, 3, 6],
            [5, 3, 3, 3, 3, 3, 3, 5],
            [1, 8, 9, 5, 6, 6, 7, 4],
        ],
        np.uint32,
    )


@pytest.fixture(name="dem_zhou_2022_filepath")
def fixture_dem_zhou_2022_filepath(dem_zhou_2022):
    """DEM from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a DEM file
    """
    dem_path = "/vsimem/dem_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = dem_zhou_2022.shape
    dataset = driver.Create(dem_path, cols, rows, 1, gdal.GDT_Float32)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset.GetRasterBand(1).WriteArray(dem_zhou_2022)
    dataset = None
    yield dem_path
    gdal.Unlink(dem_path)


@pytest.fixture(name="fdr_zhou_2022")
def fixture_fdr_zhou_2022():
    """FDR from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_EAST,
            ],
        ],
        np.uint32,
    )


@pytest.fixture(name="fdr_zhou_2022_filepath")
def fixture_fdr_zhou_2022_filepath(fdr_zhou_2022):
    """FDR from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a FDR file
    """
    fdr_path = "/vsimem/fdr_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = fdr_zhou_2022.shape
    dataset = driver.Create(fdr_path, cols, rows, 1, gdal.GDT_Byte)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(FLOW_DIRECTION_NODATA)
    dataset.GetRasterBand(1).WriteArray(fdr_zhou_2022)
    dataset = None
    yield fdr_path
    gdal.Unlink(fdr_path)


@pytest.fixture(name="expected_fixed_fdr_zhou_2022")
def fixture_expected_fixed_fdr_zhou_2022():
    """Create the expected fixed fdr for the test dem.

    Returns:
        np.ndarray: A 2D array containing the expected fixed fdr
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_EAST,
            ],
        ],
        np.uint32,
    )


@pytest.mark.parametrize("chunk_size", [2, 4, 8, 12])
def test__resolve_flats_tiled(
    dem_zhou_2022_filepath,
    fdr_zhou_2022_filepath,
    expected_fixed_fdr_zhou_2022,
    chunk_size,
):
    """
    Test the fix_flats function with different chunk sizes.
     - Chunk size 2 will contain tiles that contain only flats
     - Chunk size 4 will test the example in the paper
     - Chunk size 8 will test a single tile the size of the entire DEM
     - Chunk size 12 will test a chunk size larger than the DEM
    """
    output_filepath = f"/vsimem/fixed_fdr_{chunk_size}.tif"
    _resolve_flats_tiled(
        dem_zhou_2022_filepath,
        fdr_zhou_2022_filepath,
        output_filepath,
        chunk_size=chunk_size,
        working_dir=f"/vsimem/{chunk_size}/",
    )
    fixed_fdr_dataset = gdal.Open(output_filepath)
    fixed_fdr = fixed_fdr_dataset.GetRasterBand(1).ReadAsArray()
    flat_mask_dataset = gdal.Open(f"/vsimem/{chunk_size}/flat_mask.tif")
    flat_mask_dataset.GetRasterBand(1).ReadAsArray()
    assert np.array_equal(fixed_fdr, expected_fixed_fdr_zhou_2022)


def _run_resolve_flats_tiled(dem: np.ndarray, fdr: np.ndarray, chunk_size: int):
    """Write dem/fdr to in-memory rasters, run _resolve_flats_tiled, and
    return the fixed fdr array."""
    tag = f"seam_{id(dem)}_{chunk_size}"
    dem_path = f"/vsimem/dem_{tag}.tif"
    fdr_path = f"/vsimem/fdr_{tag}.tif"
    out_path = f"/vsimem/fixed_{tag}.tif"
    working_dir = f"/vsimem/wd_{tag}/"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = dem.shape

    dem_ds = driver.Create(dem_path, cols, rows, 1, gdal.GDT_Float32)
    dem_ds.GetRasterBand(1).SetNoDataValue(DEM_NODATA)
    dem_ds.GetRasterBand(1).WriteArray(dem)
    dem_ds = None

    fdr_ds = driver.Create(fdr_path, cols, rows, 1, gdal.GDT_Byte)
    fdr_ds.GetRasterBand(1).SetNoDataValue(FLOW_DIRECTION_NODATA)
    fdr_ds.GetRasterBand(1).WriteArray(fdr)
    fdr_ds = None

    try:
        _resolve_flats_tiled(
            dem_path, fdr_path, out_path, chunk_size=chunk_size, working_dir=working_dir
        )
        fixed_ds = gdal.Open(out_path)
        return fixed_ds.GetRasterBand(1).ReadAsArray()
    finally:
        for path in (dem_path, fdr_path, out_path):
            with contextlib.suppress(RuntimeError):
                gdal.Unlink(path)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("chunk_size", [4, 8, 16])
def test_resolve_flats_tiled_no_seam_artifacts(seed, chunk_size):
    """Random DEMs quantized to a few elevation levels produce large flats
    spanning tile seams. The tiled resolution must never produce reciprocal
    flow directions or cycles, and must resolve the same set of cells as the
    core (non-tiled) algorithm."""
    rng = np.random.default_rng(seed)
    dem = priority_flood_fill(rng.integers(0, 4, size=(24, 24)).astype(np.float32))
    initial_fdr = make_initial_fdr(dem)

    fixed_tiled = _run_resolve_flats_tiled(dem, initial_fdr, chunk_size)

    assert find_reciprocal_pairs(fixed_tiled) == []
    assert_valid_drainage(fixed_tiled)

    # the tiled algorithm must drain the same cells as the core algorithm
    # (exact directions may differ at seams, drainage-equivalence matters)
    fixed_core = fix_flats(dem, initial_fdr.copy(), inplace=False)
    np.testing.assert_array_equal(
        fixed_tiled == FLOW_DIRECTION_UNDEFINED,
        fixed_core == FLOW_DIRECTION_UNDEFINED,
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("shape", [(8, 48), (48, 8)], ids=["1xN grid", "Nx1 grid"])
def test_resolve_flats_tiled_degenerate_tile_grid(seed, shape):
    """Seams must be joined when the tile grid is a single tile row or column.

    Regression test. Joining seams by walking a 2x2 tile block over
    range(rows - 1) x range(cols - 1) iterates zero times on a 1xN or Nx1 tile
    grid, leaving every seam unjoined so flats straddling a tile boundary are
    resolved as if they ended there. The shapes here are smaller than chunk_size
    in one dimension and larger in the other, which is exactly the case the block
    walk misses.
    """
    rng = np.random.default_rng(seed)
    dem = priority_flood_fill(rng.integers(0, 4, size=shape).astype(np.float32))
    initial_fdr = make_initial_fdr(dem)

    fixed_tiled = _run_resolve_flats_tiled(dem, initial_fdr, chunk_size=8)

    assert find_reciprocal_pairs(fixed_tiled) == []
    assert_valid_drainage(fixed_tiled)

    fixed_core = fix_flats(dem, initial_fdr.copy(), inplace=False)
    np.testing.assert_array_equal(
        fixed_tiled == FLOW_DIRECTION_UNDEFINED,
        fixed_core == FLOW_DIRECTION_UNDEFINED,
    )


def test_seam_scale_mismatch_regression():
    """Regression for a real-data bug: per-tile flat masks are on
    non-comparable numeric scales across a tile seam (the neighboring tile
    participated in the away-from-higher pass, +MAX_FLAT_HEIGHT, this tile
    did not). Comparing raw masks through the halo makes the true cross-seam
    exit look uphill, and the first-zero-slope tie-break then points two
    vertically adjacent seam cells at each other, creating a two-cell cycle
    in the output flow directions. The seam-aware variant compares canonical masks
    reconstructed from the globally solved distances and must not cycle."""
    chunk = 4
    size = chunk + 2  # buffer-1 tile read

    # one flat at elevation 5 filling the tile and the west halo; north,
    # south, and east halos are higher ground
    dem = np.full((size, size), 5.0, dtype=np.float32)
    dem[0, :] = 9.0
    dem[-1, :] = 9.0
    dem[:, -1] = 9.0

    fdr = np.where(dem == 5.0, FLOW_DIRECTION_UNDEFINED, FLOW_DIRECTION_EAST).astype(
        np.uint8
    )

    # raw per-tile masks as the buggy bookkeeping produces them: this tile is
    # on the small towards-lower-only scale, pulling flow west toward the
    # seam; the west halo (from the neighbor tile) is on the huge
    # away-from-higher scale
    flat_mask = np.zeros((size, size), dtype=np.int64)
    for col in range(1, size - 1):
        flat_mask[1:-1, col] = 2 * (col + 1)
    flat_mask[:, 0] = MAX_FLAT_HEIGHT - 4 + 2 * 1

    # current behavior: the west halo looks uphill, every interior seam-column
    # cell falls back to a zero-slope vertical neighbor, and the two top seam
    # cells select each other
    fdr_raw = fdr.copy()
    d8_masked_flow_dirs(dem, flat_mask, fdr_raw)
    interior_raw = fdr_raw[1:-1, 1:-1]
    assert ((1, 0), (0, 0)) in find_reciprocal_pairs(interior_raw) or (
        (0, 0),
        (1, 0),
    ) in find_reciprocal_pairs(interior_raw)

    # globally consistent solved distances: the flat's low edge lies west of
    # the seam, so the west halo cells are strictly closer to it
    tile_rows, tile_cols = 1, 2
    perimeter_len = 4 * chunk - 4
    dist_high = np.zeros((2, perimeter_len), dtype=np.int64)
    dist_low = np.zeros((2, perimeter_len), dtype=np.int64)
    west = Int64Perimeter(dist_low[0], chunk, chunk, 0)
    own = Int64Perimeter(dist_low[1], chunk, chunk, 1)
    for i in range(chunk):
        j = west.get_index(i, chunk - 1)
        dist_high[0][j] = 4
        dist_low[0][j] = 1
    for i in range(perimeter_len):
        row, col = own.get_row_col(i)
        dist_high[1][i] = 5 + col
        dist_low[1][i] = 2 + col

    seam = build_seam_canonical_mask(
        dist_high, dist_low, 0, 1, tile_rows, tile_cols, chunk
    )
    fdr_seam = fdr.copy()
    d8_masked_flow_dirs_seam(dem, flat_mask, fdr_seam, seam)
    interior_seam = fdr_seam[1:-1, 1:-1]

    assert find_reciprocal_pairs(interior_seam) == []
    # the seam column now drains west across the seam as it should
    np.testing.assert_array_equal(
        interior_seam[:, 0], np.full(chunk, FLOW_DIRECTION_WEST, dtype=np.uint8)
    )


def test_canonical_mask_cases():
    """_canonical_mask must mirror the four cases of towards_lower_tile."""
    assert _canonical_mask(0, 0) == 0
    assert _canonical_mask(0, 3) == 6  # towards-lower only: 2 * dist
    assert _canonical_mask(5, 0) == -5  # away-from-higher only: -dist
    assert _canonical_mask(5, 3) == MAX_FLAT_HEIGHT - 5 + 2 * 3


def test_build_seam_canonical_mask_index_mapping():
    """The seam mask must place each neighbor tile's perimeter values at the
    correct halo positions of a buffer-1 tile read."""
    chunk = 4
    tile_rows, tile_cols = 1, 2
    perimeter_len = 4 * chunk - 4
    dist_high = np.zeros((2, perimeter_len), dtype=np.int64)
    dist_low = np.zeros((2, perimeter_len), dtype=np.int64)

    # give every perimeter cell of both tiles a distinct low-edge distance
    dist_low[0] = np.arange(1, perimeter_len + 1)
    dist_low[1] = np.arange(101, perimeter_len + 101)

    seam = build_seam_canonical_mask(
        dist_high, dist_low, 0, 1, tile_rows, tile_cols, chunk
    )

    # own ring of tile (0, 1): every perimeter cell at its (row+1, col+1)
    own = Int64Perimeter(dist_low[1], chunk, chunk, 1)
    for i in range(perimeter_len):
        row, col = own.get_row_col(i)
        assert seam[row + 1, col + 1] == 2 * dist_low[1][i]

    # west halo of tile (0, 1): the west tile's east column, top to bottom
    west = Int64Perimeter(dist_low[0], chunk, chunk, 0)
    for i in range(chunk):
        j = west.get_index(i, chunk - 1)
        assert seam[i + 1, 0] == 2 * dist_low[0][j]

    # no tile east of tile (0, 1): halo stays unknown
    assert np.all(seam[:, chunk + 1] == SEAM_MASK_UNKNOWN)
    # interior cells are not populated
    assert np.all(seam[2:chunk, 2:chunk] == SEAM_MASK_UNKNOWN)
