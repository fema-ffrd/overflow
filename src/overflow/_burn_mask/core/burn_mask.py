import numpy as np
from numba import boolean, float64, njit, prange  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import Dict  # type: ignore[attr-defined]
from numba.types import int64 as nb_int64

from overflow._util.constants import (
    BURN_METHOD_CONSTANT,
    BURN_METHOD_RELATIVE,
    BURN_METHOD_STATISTIC,
    BURN_NO_REGION_LABEL,
    BURN_STAT_MAX,
    BURN_STAT_MEAN,
    BURN_STAT_MIN,
    NEIGHBOR_OFFSETS_4,
    NEIGHBOR_OFFSETS_8,
)
from overflow._util.queue import Int64PairQueue
from overflow._util.raster import create_dataset, open_dataset

# number of columns in the per region statistics array: min, max, sum, count
REGION_STAT_COLUMNS = 4
# first label used by the in memory path; BURN_NO_REGION_LABEL is reserved
CORE_LABEL_OFFSET = 1
REGION_STAT_MIN = 0
REGION_STAT_MAX = 1
REGION_STAT_SUM = 2
REGION_STAT_COUNT = 3

_BURN_METHODS = {
    "constant": BURN_METHOD_CONSTANT,
    "relative": BURN_METHOD_RELATIVE,
    "statistic": BURN_METHOD_STATISTIC,
}
_BURN_STATISTICS = {
    "min": BURN_STAT_MIN,
    "max": BURN_STAT_MAX,
    "mean": BURN_STAT_MEAN,
}


@jitclass(
    [
        ("values", float64[:]),
        ("burns", float64[:]),
        ("select_all", boolean),
        ("nodata", float64),
        ("has_nodata", boolean),
        ("broadcast_burn", float64),
        ("has_broadcast", boolean),
    ]
)
class MaskSelection:
    """
    Describes which mask raster values identify regions to burn, and for the
    constant and relative methods how much to burn for each of those values.

    Selection is by exact equality against the mask value, so classified masks
    should hold exact class values. Mask nodata and NaN are never selected.

    Attributes:
        values (float64[:]): The mask values that identify regions. Empty when
            select_all is True.
        burns (float64[:]): The burn value for each entry of values, in the same
            order. Empty for the statistic method or when has_broadcast is True.
        select_all (boolean): If True, every non zero, non nodata mask value
            identifies regions. Each distinct value still forms its own regions.
        nodata (float64): The mask raster's nodata value.
        has_nodata (boolean): Whether the mask raster declares a nodata value.
        broadcast_burn (float64): A single burn value applied to every selected
            mask value.
        has_broadcast (boolean): Whether broadcast_burn should be used in place of
            the per value burns array.
    """

    def __init__(
        self,
        values: np.ndarray,
        burns: np.ndarray,
        select_all: bool,
        nodata: float,
        has_nodata: bool,
        broadcast_burn: float,
        has_broadcast: bool,
    ):
        self.values = values
        self.burns = burns
        self.select_all = select_all
        self.nodata = nodata
        self.has_nodata = has_nodata
        self.broadcast_burn = broadcast_burn
        self.has_broadcast = has_broadcast

    def is_selected(self, value) -> bool:
        """Return True if a mask value identifies a region to burn.

        Args:
            value: A cell value from the mask raster.

        Returns:
            bool: True if the value identifies a region.
        """
        # value != value is a dtype agnostic NaN check; numba rejects np.isnan on ints
        if value != value:
            return False
        if self.has_nodata and value == self.nodata:
            return False
        if self.select_all:
            return bool(value != 0)
        # an explicit loop rather than any(): numba does not compile generator
        # expressions inside jitclass methods
        for i in range(self.values.shape[0]):  # noqa: SIM110
            if value == self.values[i]:
                return True
        return False

    def burn_for(self, value) -> float:
        """Return the burn value configured for a selected mask value.

        Only meaningful for the constant and relative methods. Callers must have
        checked is_selected first.

        Args:
            value: A cell value from the mask raster.

        Returns:
            float: The configured burn value, or NaN if the value has none.
        """
        if self.has_broadcast:
            return self.broadcast_burn
        for i in range(self.values.shape[0]):
            if value == self.values[i]:
                return float(self.burns[i])
        return np.nan


@njit
def label_regions(
    mask: np.ndarray,
    selection: MaskSelection,
    label_offset: int,
    neighbor_offsets,
    valid_rows: int,
    valid_cols: int,
) -> tuple:
    """Label each contiguous region of the mask with a unique integer.

    A region is a maximal connected set of cells that share the same mask value and
    whose value is selected. Cells of two different selected values never join into
    one region even when they touch. This is a flood fill in the shape of
    Algorithm 4 LabelFlats (https://rbarnes.org/sci/2014_flats.pdf), with equality
    of mask value standing in for equality of elevation.

    Labels are assigned from label_offset upwards. Callers processing tiles pass a
    per tile offset so that labels are globally unique without a renumbering pass.

    Tiles that hang off the right or bottom edge of the raster are padded out to a
    full square by the reader. Those padded cells must not be labeled: if they were,
    two genuinely disconnected regions on either side of a tile seam could be joined
    through padding that shares their mask value. valid_rows and valid_cols bound
    the real data.

    Args:
        mask (np.ndarray): The mask raster tile.
        selection (MaskSelection): Which mask values identify regions.
        label_offset (int): The first label to assign.
        neighbor_offsets (np.ndarray): Row/column offsets defining connectivity,
            either NEIGHBOR_OFFSETS_4 or NEIGHBOR_OFFSETS_8.
        valid_rows (int): Number of rows of the tile that hold real raster data.
        valid_cols (int): Number of columns of the tile that hold real raster data.

    Returns:
        tuple: (labels, region_count) where labels is an int64 array the same shape
            as mask holding BURN_NO_REGION_LABEL outside regions, and region_count
            is the number of regions labeled.
    """
    n_row, n_col = mask.shape
    n_row = min(n_row, valid_rows)
    n_col = min(n_col, valid_cols)
    labels = np.full(mask.shape, BURN_NO_REGION_LABEL, dtype=np.int64)
    region_count = 0
    # one queue reused across regions; allocating per region is wasteful when the
    # mask holds many small regions
    to_fill = Int64PairQueue([(np.int64(0), np.int64(0))])
    to_fill.pop()
    for row in range(n_row):
        for col in range(n_col):
            if labels[row, col] != BURN_NO_REGION_LABEL:
                continue
            value = mask[row, col]
            if not selection.is_selected(value):
                continue
            new_label = np.int64(label_offset + region_count)
            region_count += 1
            to_fill.push((np.int64(row), np.int64(col)))
            while len(to_fill) > 0:
                fill_row, fill_col = to_fill.pop()
                not_in_bounds = (
                    fill_row < 0
                    or fill_row >= n_row
                    or fill_col < 0
                    or fill_col >= n_col
                )
                if not_in_bounds:
                    continue
                if labels[fill_row, fill_col] != BURN_NO_REGION_LABEL:
                    continue
                if mask[fill_row, fill_col] != value:
                    continue
                labels[fill_row, fill_col] = new_label
                for k in range(neighbor_offsets.shape[0]):
                    to_fill.push(
                        (
                            np.int64(fill_row + neighbor_offsets[k, 0]),
                            np.int64(fill_col + neighbor_offsets[k, 1]),
                        )
                    )
    return labels, region_count


@njit
def accumulate_region_stats(
    dem: np.ndarray,
    labels: np.ndarray,
    label_offset: int,
    region_count: int,
    dem_nodata: float,
) -> np.ndarray:
    """Accumulate min, max, sum and count of the DEM under each labeled region.

    These four quantities are all mergeable: the statistics of a region split
    across tiles are recovered by taking the min of the mins, the max of the maxes
    and the sums of the sums and counts. That is what lets min, max and mean be
    computed exactly for regions that straddle tile boundaries.

    DEM nodata and NaN cells are excluded, so a region lying entirely over DEM
    nodata ends up with a count of zero.

    Args:
        dem (np.ndarray): The DEM tile.
        labels (np.ndarray): Region labels for the same tile.
        label_offset (int): The first label used by this tile.
        region_count (int): The number of regions labeled in this tile.
        dem_nodata (float): The DEM's nodata value.

    Returns:
        np.ndarray: A float64 array of shape (region_count, 4) holding
            (min, max, sum, count) for each region, ordered by label.
    """
    stats = np.zeros((region_count, REGION_STAT_COLUMNS), dtype=np.float64)
    stats[:, REGION_STAT_MIN] = np.inf
    stats[:, REGION_STAT_MAX] = -np.inf
    n_row, n_col = dem.shape
    for row in range(n_row):
        for col in range(n_col):
            label = labels[row, col]
            if label == BURN_NO_REGION_LABEL:
                continue
            elevation = dem[row, col]
            if elevation == dem_nodata or np.isnan(elevation):
                continue
            index = label - label_offset
            value = np.float64(elevation)
            if value < stats[index, REGION_STAT_MIN]:
                stats[index, REGION_STAT_MIN] = value
            if value > stats[index, REGION_STAT_MAX]:
                stats[index, REGION_STAT_MAX] = value
            stats[index, REGION_STAT_SUM] += value
            stats[index, REGION_STAT_COUNT] += 1.0
    return stats


@njit
def region_value(
    minimum: float,
    maximum: float,
    total: float,
    count: float,
    statistic: int,
    burn_offset: float,
) -> float:
    """Reduce a region's accumulated statistics to the elevation to burn.

    Args:
        minimum (float): The region's minimum DEM value.
        maximum (float): The region's maximum DEM value.
        total (float): The sum of the region's DEM values.
        count (float): The number of valid DEM cells in the region.
        statistic (int): One of BURN_STAT_MIN, BURN_STAT_MAX, BURN_STAT_MEAN.
        burn_offset (float): Subtracted from the computed statistic.

    Returns:
        float: The elevation to burn, or NaN if the region has no valid DEM
            cells and should be left untouched.
    """
    if count == 0.0:
        return np.nan
    if statistic == BURN_STAT_MIN:
        value = minimum
    elif statistic == BURN_STAT_MAX:
        value = maximum
    else:
        value = total / count
    return value - burn_offset


@njit
def build_burn_lookup(
    stats: np.ndarray, label_offset: int, statistic: int, burn_offset: float
):
    """Build a label to burn value lookup from a single array of region statistics.

    Used by the in memory path, where every region is already whole. The tiled path
    builds its lookup in GlobalState.solve_regions instead, after merging regions
    that span tiles.

    Args:
        stats (np.ndarray): Region statistics from accumulate_region_stats.
        label_offset (int): The first label the statistics correspond to.
        statistic (int): One of BURN_STAT_MIN, BURN_STAT_MAX, BURN_STAT_MEAN.
        burn_offset (float): Subtracted from each computed statistic.

    Returns:
        Dict: A numba typed dict mapping each label to its burn value. Regions with
            no valid DEM cells map to NaN.
    """
    lookup = Dict.empty(key_type=nb_int64, value_type=float64)
    for index in range(stats.shape[0]):
        lookup[np.int64(label_offset + index)] = region_value(
            stats[index, REGION_STAT_MIN],
            stats[index, REGION_STAT_MAX],
            stats[index, REGION_STAT_SUM],
            stats[index, REGION_STAT_COUNT],
            statistic,
            burn_offset,
        )
    return lookup


@njit(nogil=True, parallel=True)
def burn_direct(
    dem: np.ndarray,
    mask: np.ndarray,
    selection: MaskSelection,
    method: int,
    dem_nodata: float,
    tile_row: int,
    tile_col: int,
) -> tuple:
    """Burn constant or relative values into a DEM tile, in place.

    Both methods are pure per cell functions of the DEM value and the mask value,
    so no region labeling is needed and the tiled driver can run them in a single
    pass with no cross tile reconciliation.

    Args:
        dem (np.ndarray): The DEM tile, modified in place.
        mask (np.ndarray): The mask tile.
        selection (MaskSelection): Which mask values to burn and by how much.
        method (int): BURN_METHOD_CONSTANT or BURN_METHOD_RELATIVE.
        dem_nodata (float): The DEM's nodata value.
        tile_row (int): The tile's row in the tile grid, passed through to the
            caller so results can be routed without tracking futures.
        tile_col (int): The tile's column in the tile grid.

    Returns:
        tuple: (dem, tile_row, tile_col) with dem modified in place.
    """
    n_row, n_col = dem.shape
    for row in prange(n_row):
        for col in range(n_col):
            elevation = dem[row, col]
            if elevation == dem_nodata or np.isnan(elevation):
                continue
            value = mask[row, col]
            if not selection.is_selected(value):
                continue
            burn = selection.burn_for(value)
            if np.isnan(burn):
                continue
            if method == BURN_METHOD_CONSTANT:
                dem[row, col] = burn
            else:
                dem[row, col] = elevation - burn
    return dem, tile_row, tile_col


@njit(nogil=True, parallel=True)
def apply_burn(
    dem: np.ndarray,
    labels: np.ndarray,
    burn_lookup,
    dem_nodata: float,
    tile_row: int,
    tile_col: int,
) -> tuple:
    """Write each region's solved burn value into a DEM tile, in place.

    Args:
        dem (np.ndarray): The DEM tile, modified in place.
        labels (np.ndarray): Region labels for the same tile.
        burn_lookup: Numba typed dict mapping every label present in labels to its
            burn value. Labels mapping to NaN are left untouched.
        dem_nodata (float): The DEM's nodata value.
        tile_row (int): The tile's row in the tile grid, passed through to the
            caller so results can be routed without tracking futures.
        tile_col (int): The tile's column in the tile grid.

    Returns:
        tuple: (dem, tile_row, tile_col) with dem modified in place.
    """
    n_row, n_col = dem.shape
    for row in prange(n_row):
        for col in range(n_col):
            label = labels[row, col]
            if label == BURN_NO_REGION_LABEL:
                continue
            elevation = dem[row, col]
            if elevation == dem_nodata or np.isnan(elevation):
                continue
            # every label written by label_regions is present in the lookup by
            # construction, so this cannot miss
            burn = burn_lookup[label]
            if np.isnan(burn):
                continue
            dem[row, col] = burn
    return dem, tile_row, tile_col


def parse_burn_method(method: str) -> int:
    """Translate the public burn method string into its integer flag.

    Args:
        method (str): One of "constant", "relative", "statistic".

    Returns:
        int: The matching BURN_METHOD_* constant.

    Raises:
        ValueError: If the method is not recognized.
    """
    try:
        return _BURN_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown burn method '{method}', expected one of: "
            f"{', '.join(_BURN_METHODS)}"
        ) from None


def parse_statistic(statistic: str) -> int:
    """Translate the public statistic string into its integer flag.

    Args:
        statistic (str): One of "min", "max", "mean".

    Returns:
        int: The matching BURN_STAT_* constant.

    Raises:
        ValueError: If the statistic is not recognized.
    """
    try:
        return _BURN_STATISTICS[statistic]
    except KeyError:
        raise ValueError(
            f"Unknown statistic '{statistic}', expected one of: "
            f"{', '.join(_BURN_STATISTICS)}"
        ) from None


def parse_connectivity(connectivity: int) -> np.ndarray:
    """Translate a connectivity setting into its neighbor offsets.

    Args:
        connectivity (int): Either 4 or 8.

    Returns:
        np.ndarray: NEIGHBOR_OFFSETS_4 or NEIGHBOR_OFFSETS_8.

    Raises:
        ValueError: If the connectivity is neither 4 nor 8.
    """
    if connectivity == 8:
        return NEIGHBOR_OFFSETS_8
    if connectivity == 4:
        return NEIGHBOR_OFFSETS_4
    raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")


def parse_burn_values(burn_values: str | float | dict) -> dict | float:
    """Parse the burn values argument into a mapping or a single broadcast value.

    Accepts the CLI's paired mapping string ("1:225.5,3:210.0"), a bare number
    string applied to every selected mask value ("1.5"), or the already parsed
    Python forms of either.

    Args:
        burn_values (str | float | dict): The value to parse.

    Returns:
        dict | float: A mask value to burn value mapping, or a single float to
            broadcast to every selected mask value.

    Raises:
        ValueError: If a mapping entry is malformed or holds a non numeric value.
    """
    if isinstance(burn_values, dict):
        return {float(key): float(value) for key, value in burn_values.items()}
    if not isinstance(burn_values, str):
        return float(burn_values)
    text = burn_values.strip()
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"Could not parse burn values '{burn_values}', expected a number "
                "or a mapping like '1:225.5,3:210.0'"
            ) from None
    parsed = {}
    for entry in text.split(","):
        if not entry.strip():
            continue
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Could not parse burn value entry '{entry}', expected 'value:burn'"
            )
        try:
            parsed[float(parts[0])] = float(parts[1])
        except ValueError:
            raise ValueError(
                f"Burn value entry '{entry}' has a non numeric part"
            ) from None
    if not parsed:
        raise ValueError(f"Burn values '{burn_values}' contained no entries")
    return parsed


def parse_mask_values(mask_values: str | list | tuple | None) -> list | None:
    """Parse the mask values argument into a list of floats.

    Args:
        mask_values (str | list | tuple | None): A comma separated string such as
            "1,3", an already parsed sequence, or None.

    Returns:
        list | None: The parsed mask values, or None if none were given.

    Raises:
        ValueError: If an entry is not numeric or the string holds no entries.
    """
    if mask_values is None:
        return None
    if not isinstance(mask_values, str):
        return [float(value) for value in mask_values]
    entries = [entry.strip() for entry in mask_values.split(",") if entry.strip()]
    if not entries:
        raise ValueError(f"Mask values '{mask_values}' contained no entries")
    try:
        return [float(entry) for entry in entries]
    except ValueError:
        raise ValueError(
            f"Could not parse mask values '{mask_values}', expected a comma "
            "separated list of numbers like '1,3'"
        ) from None


def build_mask_selection(
    method: int,
    mask_values: str | list | tuple | None,
    burn_values: str | float | dict | None,
    mask_nodata: float | None,
) -> MaskSelection:
    """Resolve the mask value and burn value arguments into a MaskSelection.

    When mask_values is omitted, the mapping keys of burn_values supply it. When
    neither names specific values, every non zero, non nodata mask value is
    selected and each distinct value still forms its own regions.

    Args:
        method (int): One of the BURN_METHOD_* constants.
        mask_values (str | list | tuple | None): Mask values identifying regions.
        burn_values (str | float | dict | None): Burn values for the constant and
            relative methods.
        mask_nodata (float | None): The mask raster's nodata value, if it has one.

    Returns:
        MaskSelection: The resolved selection.

    Raises:
        ValueError: If burn values are missing or incomplete for the constant and
            relative methods, or if a selected mask value is the mask's nodata.
    """
    values = parse_mask_values(mask_values)
    needs_burn_values = method in (BURN_METHOD_CONSTANT, BURN_METHOD_RELATIVE)

    parsed_burns: dict | float | None = None
    if burn_values is not None:
        parsed_burns = parse_burn_values(burn_values)
    elif needs_burn_values:
        raise ValueError(
            "burn_values is required for the constant and relative methods"
        )

    if values is None and isinstance(parsed_burns, dict):
        values = sorted(parsed_burns)

    if values is not None and mask_nodata is not None:
        conflicting = [value for value in values if value == mask_nodata]
        if conflicting:
            raise ValueError(
                f"Mask value {conflicting[0]} is the mask raster's nodata value and "
                "cannot identify a region"
            )

    burns_array = np.empty(0, dtype=np.float64)
    broadcast_burn = np.nan
    has_broadcast = False
    if needs_burn_values:
        if isinstance(parsed_burns, dict):
            assert values is not None
            missing = [value for value in values if value not in parsed_burns]
            if missing:
                raise ValueError(
                    f"No burn value given for mask value(s) "
                    f"{', '.join(str(value) for value in missing)}"
                )
            burns_array = np.array(
                [parsed_burns[value] for value in values], dtype=np.float64
            )
        else:
            broadcast_burn = float(parsed_burns)  # type: ignore[arg-type]
            has_broadcast = True

    return MaskSelection(
        np.array(values if values is not None else [], dtype=np.float64),
        burns_array,
        values is None,
        float(mask_nodata) if mask_nodata is not None else np.nan,
        mask_nodata is not None,
        broadcast_burn,
        has_broadcast,
    )


def resolve_burn_arguments(
    mask_path: str,
    method: str,
    mask_values: str | list | tuple | None,
    burn_values: str | float | dict | None,
    statistic: str,
    connectivity: int,
) -> tuple:
    """Validate and resolve the public burn arguments into their internal forms.

    Shared by the in memory and tiled paths so both agree on what the arguments
    mean, and so every argument is checked before any raster work starts.

    Args:
        mask_path (str): Path to the mask raster, opened to read its nodata value.
        method (str): One of "constant", "relative", "statistic".
        mask_values (str | list | tuple | None): Mask values identifying regions.
        burn_values (str | float | dict | None): Burn values for the constant and
            relative methods.
        statistic (str): One of "min", "max", "mean".
        connectivity (int): Either 4 or 8.

    Returns:
        tuple: (method_flag, selection, statistic_flag, neighbor_offsets).

    Raises:
        ValueError: If any argument is unrecognized, malformed or inconsistent.
    """
    method_flag = parse_burn_method(method)
    statistic_flag = parse_statistic(statistic)
    neighbor_offsets = parse_connectivity(connectivity)
    mask_ds = open_dataset(mask_path)
    mask_nodata = mask_ds.GetRasterBand(1).GetNoDataValue()
    mask_ds = None
    selection = build_mask_selection(method_flag, mask_values, burn_values, mask_nodata)
    return method_flag, selection, statistic_flag, neighbor_offsets


def _burn_mask_core(
    dem_path: str,
    mask_path: str,
    output_path: str,
    method: int,
    selection: MaskSelection,
    statistic: int,
    burn_offset: float,
    neighbor_offsets: np.ndarray,
) -> None:
    """
    Burn mask regions into a DEM entirely in memory.

    Suitable for DEMs that fit in RAM. Region labeling sees the whole raster at
    once, so no cross tile reconciliation is needed.

    Parameters
    ----------
    dem_path : str
        Path to the input DEM raster.
    mask_path : str
        Path to the co-registered mask raster.
    output_path : str
        Path to the output burned DEM raster.
    method : int
        One of the BURN_METHOD_* constants.
    selection : MaskSelection
        Which mask values identify regions and, for the constant and relative
        methods, how much to burn for each.
    statistic : int
        One of the BURN_STAT_* constants. Only used by the statistic method.
    burn_offset : float
        Subtracted from each computed statistic. Only used by the statistic method.
    neighbor_offsets : np.ndarray
        Row/column offsets defining region connectivity.
    """
    dem_ds = open_dataset(dem_path)
    mask_ds = open_dataset(mask_path)
    dem_band = dem_ds.GetRasterBand(1)
    mask_band = mask_ds.GetRasterBand(1)
    dem_nodata = dem_band.GetNoDataValue()
    if dem_nodata is None:
        raise ValueError("Input DEM must have a no data value")

    dem = dem_band.ReadAsArray()
    mask = mask_band.ReadAsArray()

    if method == BURN_METHOD_STATISTIC:
        labels, region_count = label_regions(
            mask,
            selection,
            CORE_LABEL_OFFSET,
            neighbor_offsets,
            mask.shape[0],
            mask.shape[1],
        )
        stats = accumulate_region_stats(
            dem, labels, CORE_LABEL_OFFSET, region_count, dem_nodata
        )
        burn_lookup = build_burn_lookup(
            stats, CORE_LABEL_OFFSET, statistic, burn_offset
        )
        dem, _, _ = apply_burn(dem, labels, burn_lookup, dem_nodata, 0, 0)
    else:
        dem, _, _ = burn_direct(dem, mask, selection, method, dem_nodata, 0, 0)

    output_ds = create_dataset(
        output_path,
        dem_nodata,
        dem_band.DataType,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(dem)
    output_band.FlushCache()
    output_ds.FlushCache()
    output_band = None
    output_ds = None
    dem_band = None
    mask_band = None
    dem_ds = None
    mask_ds = None
