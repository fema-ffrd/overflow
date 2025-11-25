"""
Overflow - High-performance Python library for hydrological terrain analysis.

- breach: Breach pits using least-cost paths
- fill: Fill depressions in a DEM
- flow_direction: Compute D8 flow directions and resolve flats
- accumulation: Calculate flow accumulation
- basins: Delineate drainage basins from drainage points
- streams: Extract stream networks
- flow_length: Calculate upstream flow length (longest flow path)
"""

from overflow._basins.core import (
    _drainage_points_from_file,
    _label_watersheds_core,
)
from overflow._basins.core import (
    update_drainage_points_file as _update_drainage_points_file,
)
from overflow._basins.tiled import _label_watersheds_tiled
from overflow._breach_paths_least_cost import _breach_paths_least_cost
from overflow._extract_streams.core import _extract_streams_core
from overflow._extract_streams.tiled import _extract_streams_tiled
from overflow._fill_depressions.core import _fill_depressions
from overflow._fill_depressions.tiled import _fill_depressions_tiled
from overflow._flow_accumulation.core import _flow_accumulation
from overflow._flow_accumulation.tiled import _flow_accumulation_tiled
from overflow._flow_direction import _flow_direction
from overflow._longest_flow_path import _flow_length_core
from overflow._resolve_flats.core import _resolve_flats_core
from overflow._resolve_flats.tiled import _resolve_flats_tiled
from overflow._util.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_RADIUS
from overflow._util.progress import ProgressCallback
from overflow._util.raster import snap_drainage_points as _snap_drainage_points
from overflow.codes import FlowDirection

__version__ = "0.3.4"


def breach(
    input_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_cost: float = float("inf"),
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Breach pits in a DEM using least-cost paths.

    This function identifies pits (local minima) in the DEM and creates breach
    paths to allow water to flow out, minimizing the total elevation change.

    Args:
        input_path: Path to the input DEM raster file.
        output_path: Path for the output breached DEM raster file.
        chunk_size: Size of processing chunks in pixels. Default is 2048.
        search_radius: Maximum search radius for finding breach paths in cells.
            Default is 50.
        max_cost: Maximum allowed cost (total elevation change) for breach paths.
            Default is infinity (no limit).
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
    """
    _breach_paths_least_cost(
        input_path, output_path, chunk_size, search_radius, max_cost, progress_callback
    )


def fill(
    input_path: str,
    output_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str | None = None,
    fill_holes: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Fill depressions in a DEM using priority flood algorithm.

    This function fills local depressions (sinks) in the DEM to create a
    hydrologically conditioned surface where water can flow to the edges.

    Args:
        input_path: Path to the input DEM raster file.
        output_path: Path for the output filled DEM raster file.
            If None, the input file is modified in place.
        chunk_size: Size of processing chunks in pixels. Use chunk_size <= 1 for
            in-memory processing (suitable for smaller DEMs). Default is 2048.
        working_dir: Directory for temporary files during tiled processing.
            If None, uses system temp directory.
        fill_holes: If True, also fills holes (nodata regions) in the DEM.
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
    """
    if chunk_size <= 1:
        _fill_depressions(input_path, output_path, fill_holes)
    else:
        _fill_depressions_tiled(
            input_path,
            output_path,
            chunk_size,
            working_dir,
            fill_holes,
            progress_callback,
        )


def flow_direction(
    input_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str | None = None,
    resolve_flats: bool = True,
    progress_callback: ProgressCallback | None = None,
    flat_resolution_chunk_size_max: int = 512,
) -> None:
    """
    Compute D8 flow directions from a DEM and optionally resolve flat areas.

    This function calculates the steepest descent direction for each cell using
    the D8 algorithm, then optionally resolves flat areas to ensure continuous
    flow paths.

    Args:
        input_path: Path to the input DEM raster file (should be hydrologically
            conditioned, e.g., filled or breached).
        output_path: Path for the output flow direction raster file.
        chunk_size: Size of processing chunks in pixels. Use chunk_size <= 1 for
            in-memory processing. Default is 2048.
        working_dir: Directory for temporary files during tiled processing.
            If None, uses system temp directory.
        resolve_flats: If True (default), resolve flat areas in the flow direction
            raster to ensure water flows toward lower terrain.
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
        flat_resolution_chunk_size_max: Maximum chunk size for flat resolution processing.
            Default is 512. This caps the chunk size used during flat resolution to prevent
            performance issues in areas with large undefined flow regions. This is an advanced
            parameter only available in the Python API. When chunk_size exceeds this value,
            flat resolution will use this smaller chunk size instead
    """
    # Compute initial flow directions
    _flow_direction(input_path, output_path, chunk_size, progress_callback)

    # Resolve flat areas if requested
    if resolve_flats:
        if chunk_size <= 1:
            _resolve_flats_core(input_path, output_path, None)
        else:
            _resolve_flats_tiled(
                input_path,
                output_path,
                None,
                min(chunk_size, flat_resolution_chunk_size_max),
                working_dir,
                progress_callback,
            )


def accumulation(
    input_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Calculate flow accumulation from a flow direction raster.

    This function computes the number of upstream cells that flow into each
    cell, representing drainage area in cell units.

    Args:
        input_path: Path to the input flow direction raster file.
        output_path: Path for the output flow accumulation raster file.
        chunk_size: Size of processing chunks in pixels. Use chunk_size <= 1 for
            in-memory processing. Default is 2048.
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
    """
    if chunk_size <= 1:
        _flow_accumulation(input_path, output_path)
    else:
        _flow_accumulation_tiled(input_path, output_path, chunk_size, progress_callback)


def basins(
    fdr_path: str,
    drainage_points_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    all_basins: bool = False,
    fac_path: str | None = None,
    snap_radius: int = 0,
    layer_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Delineate drainage basins from a flow direction raster and drainage points.

    This function labels each cell with the ID of its downstream drainage point,
    effectively delineating basin boundaries.

    Args:
        fdr_path: Path to the input flow direction raster file.
        drainage_points_path: Path to the drainage points vector file.
        output_path: Path for the output basins raster file.
        chunk_size: Size of processing chunks in pixels. Use chunk_size <= 1 for
            in-memory processing. Default is 2048.
        all_basins: If True, label all basins including those not draining to
            specified points. Default is False.
        fac_path: Path to flow accumulation raster for snapping drainage points.
            If None, no snapping is performed.
        snap_radius: Radius in cells to search for nearest drainage point.
            If 0, no snapping is performed.
        layer_name: Name of the layer in the drainage points file to use.
            If None, uses the first layer.
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
    """
    drainage_points, fid_mapping = _drainage_points_from_file(
        fdr_path, drainage_points_path, layer_name
    )
    # Snap drainage points to flow accumulation grid if fac_file is provided
    if fac_path is not None and snap_radius > 0:
        drainage_points, fid_mapping = _snap_drainage_points(
            drainage_points, fac_path, snap_radius, fid_mapping
        )
    if chunk_size <= 1:
        graph = _label_watersheds_core(
            fdr_path, drainage_points, output_path, all_basins
        )
    else:
        drainage_points, _ = _drainage_points_from_file(
            fdr_path, drainage_points_path, layer_name
        )
        graph = _label_watersheds_tiled(
            fdr_path,
            drainage_points,
            output_path,
            chunk_size,
            all_basins,
            progress_callback,
        )
    # Update drainage points file with basin_id and ds_basin_id
    _update_drainage_points_file(
        drainage_points_path, drainage_points, fid_mapping, graph, layer_name
    )


def streams(
    fac_path: str,
    fdr_path: str,
    output_dir: str,
    threshold: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Extract stream networks from flow accumulation and direction rasters.

    This function identifies stream cells based on a flow accumulation threshold
    and creates vector stream lines and junction points.

    Args:
        fac_path: Path to the input flow accumulation raster file.
        fdr_path: Path to the input flow direction raster file.
        output_dir: Directory for output stream files (streams.gpkg will be created).
        threshold: Minimum flow accumulation value (cell count) to define a stream.
            Cells with accumulation >= threshold are considered streams.
        chunk_size: Size of processing chunks in pixels. Use chunk_size <= 1 for
            in-memory processing. Default is 2048.
        progress_callback: Optional callback function for progress reporting.
            Receives a float value between 0 and 1.
    """
    if chunk_size <= 1:
        _extract_streams_core(
            fac_path, fdr_path, output_dir, threshold, progress_callback
        )
    else:
        _extract_streams_tiled(
            fac_path, fdr_path, output_dir, threshold, chunk_size, progress_callback
        )


def flow_length(
    fdr_path: str,
    drainage_points_path: str,
    output_raster: str,
    output_vector: str | None = None,
    fac_path: str | None = None,
    snap_radius: int = 0,
    layer_name: str | None = None,
) -> None:
    """
    Calculate upstream flow length (longest flow path) from drainage points.

    This function calculates the distance from each cell to its downstream
    drainage point, measured along the flow path. The cell with the maximum
    flow length in each basin represents the longest flow path origin.

    Args:
        fdr_path: Path to the input flow direction raster file.
        drainage_points_path: Path to the drainage points vector file.
        output_raster: Path for the output flow length raster file (GeoTIFF).
            Values represent upstream flow distance in map units (or meters
            for geographic CRS).
        output_vector: Path for the output longest flow path vectors (GeoPackage).
            If None, vector output is not created.
        fac_path: Path to flow accumulation raster for snapping drainage points.
            If None, no snapping is performed.
        snap_radius: Radius in cells to search for maximum flow accumulation
            when snapping drainage points. If 0 or fac_path is None, no snapping.
        layer_name: Name of the layer in the drainage points file to use.
            If None, uses the first layer.
    """
    _flow_length_core(
        fdr_path,
        drainage_points_path,
        output_raster,
        output_vector,
        layer_name,
        fac_path,
        snap_radius,
    )


__all__ = [
    # Core functions
    "breach",
    "fill",
    "flow_direction",
    "accumulation",
    "basins",
    "streams",
    "flow_length",
    # Enums
    "FlowDirection",
    # Types
    "ProgressCallback",
]
