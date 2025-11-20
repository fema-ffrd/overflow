"""
Overflow - High-performance Python library for hydrological terrain analysis.
"""

# Import core functions to make them available from the top-level package
from overflow.basins.core import drainage_points_from_file, label_watersheds_from_file
from overflow.basins.tiled import label_watersheds_tiled
from overflow.breach_paths_least_cost import breach_paths_least_cost
from overflow.breach_single_cell_pits import breach_single_cell_pits
from overflow.extract_streams.core import extract_streams
from overflow.extract_streams.tiled import extract_streams_tiled
from overflow.fill_depressions.core import fill_depressions
from overflow.fill_depressions.tiled import fill_depressions_tiled
from overflow.fix_flats.core import fix_flats_from_file
from overflow.fix_flats.tiled import fix_flats_tiled
from overflow.flow_accumulation.core import flow_accumulation
from overflow.flow_accumulation.tiled import flow_accumulation_tiled
from overflow.flow_direction import flow_direction
from overflow.util.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_RADIUS
from overflow.util.raster import feet_to_cell_count, sqmi_to_cell_count

__version__ = "0.1.10"

__all__ = [
    # Breaching
    "breach_single_cell_pits",
    "breach_paths_least_cost",
    # Depression filling
    "fill_depressions",
    "fill_depressions_tiled",
    # Flow direction
    "flow_direction",
    "fix_flats_from_file",
    "fix_flats_tiled",
    # Flow accumulation
    "flow_accumulation",
    "flow_accumulation_tiled",
    # Stream extraction
    "extract_streams",
    "extract_streams_tiled",
    # Watershed delineation
    "label_watersheds_from_file",
    "drainage_points_from_file",
    "label_watersheds_tiled",
    # Utilities
    "sqmi_to_cell_count",
    "feet_to_cell_count",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SEARCH_RADIUS",
]
