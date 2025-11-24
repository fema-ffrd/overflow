import os
import sys

import click
import numpy as np
from osgeo import gdal

from overflow import (
    __version__,
    accumulation,
    breach,
    fill,
    flow_direction,
    streams,
)
from overflow._basins.core import (
    _drainage_points_from_file,
    label_watersheds,
    update_drainage_points_file,
)
from overflow._basins.tiled import _label_watersheds_tiled
from overflow._longest_flow_path import _flow_length_core
from overflow._util.cli_progress import RichProgressDisplay
from overflow._util.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_RADIUS
from overflow._util.raster import (
    create_dataset,
    feet_to_cell_count,
    snap_drainage_points,
    sqmi_to_cell_count,
)
from overflow._util.timer import console, resource_stats, timer

# set gdal configuration
gdal.UseExceptions()
gdal.SetConfigOption("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", ""))
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))
gdal.SetConfigOption("CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE", "YES")


def print_banner():
    """Display the Overflow banner and version."""
    if sys.stdout.isatty():
        banner = f"""[bold cyan]   ╔═╗╦  ╦╔═╗╦═╗╔═╗╦  ╔═╗╦ ╦
[dim]░[/dim][cyan]▒[/cyan][bold blue]▓[/bold blue]║ ║╚╗╔╝║╣ ╠╦╝╠╣ ║  ║ ║║║║[bold blue]▓[/bold blue][cyan]▒[/cyan][dim]░[/dim]
   ╚═╝ ╚╝ ╚═╝╩╚═╚  ╩═╝╚═╝╚╩╝[/bold cyan]
        [dim]Version {__version__}[/dim]
"""
        console.print(banner)
    else:
        # Non-TTY output (plain text)
        print(f"OVERFLOW v{__version__}\n")


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """The main entry point for the command line interface."""
    print_banner()

    # If no subcommand was provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command(name="breach")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
@click.option(
    "--search_radius", help="search radius in cells", default=DEFAULT_SEARCH_RADIUS
)
@click.option(
    "--max_cost",
    help="maximum cost of breach paths (total sum of elevation removed from each cell in path)",
    default=np.inf,
)
def breach_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
    search_radius: int,
    max_cost: float,
):
    """
    Breach pits in a DEM using least-cost paths.

    This command identifies pits (local minima) in the DEM and creates breach
    paths to allow water to flow out, minimizing the total elevation change.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Breach", spinner=False):
            with progress_display.progress_context("Breaching pits"):
                breach(
                    input_file,
                    output_file,
                    chunk_size,
                    search_radius,
                    max_cost,
                    progress_display.callback,
                )
                resource_stats.add_output_file("Breached DEM", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] breach failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="fill")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff). If not provided, modifies input in place.",
    required=False,
    default=None,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 1 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--working_dir",
    help="working directory for temporary files",
)
@click.option(
    "--fill_holes",
    help="If set, fills holes (nodata regions) in the DEM",
    is_flag=True,
)
def fill_cli(
    input_file: str,
    output_file: str | None,
    chunk_size: int,
    working_dir: str | None,
    fill_holes: bool,
):
    """
    Fill depressions in a DEM using priority flood algorithm.

    This command fills local depressions (sinks) in the DEM to create a
    hydrologically conditioned surface where water can flow to the edges.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Fill", spinner=False):
            with progress_display.progress_context("Filling depressions"):
                fill(
                    input_file,
                    output_file,
                    chunk_size,
                    working_dir,
                    fill_holes,
                    progress_display.callback,
                )
                resource_stats.add_output_file("Filled DEM", output_file or input_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] fill failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="flow-direction")
@click.option(
    "--input_file",
    help="path to the DEM file",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 1 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--working_dir",
    help="working directory for temporary files",
)
@click.option(
    "--no_resolve_flats",
    help="If set, skip resolving flat areas",
    is_flag=True,
)
def flow_direction_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
    working_dir: str | None,
    no_resolve_flats: bool,
):
    """
    Compute D8 flow directions from a DEM and resolve flat areas.

    This command calculates the steepest descent direction for each cell using
    the D8 algorithm, then resolves flat areas to ensure continuous flow paths.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Flow direction", spinner=False):
            with progress_display.progress_context("Computing flow direction"):
                flow_direction(
                    input_file,
                    output_file,
                    chunk_size,
                    working_dir,
                    resolve_flats=not no_resolve_flats,
                    progress_callback=progress_display.callback,
                )
                resource_stats.add_output_file("Flow Direction", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] flow-direction failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="accumulation")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the flow direction raster",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 1 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
def accumulation_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
):
    """
    Calculate flow accumulation from a flow direction raster.

    This command computes the number of upstream cells that flow into each
    cell, representing drainage area in cell units.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Accumulation", spinner=False):
            with progress_display.progress_context("Flow Accumulation"):
                accumulation(
                    input_file, output_file, chunk_size, progress_display.callback
                )
                resource_stats.add_output_file("Flow Accumulation", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] accumulation failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="basins")
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the flow direction raster",
    required=True,
)
@click.option(
    "--fac_file",
    help="path to the GDAL supported raster dataset for the flow accumulation (optional, for snapping)",
    required=False,
    default=None,
)
@click.option(
    "--dp_file",
    help="path to the drainage points file",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 1 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--all_basins",
    help="If set, labels all basins. If not set, only labels basins upstream of drainage points.",
    default=False,
)
@click.option(
    "--dp_layer",
    help="name of the layer in the drainage points file",
    required=False,
    default=None,
)
@click.option(
    "--snap_radius_ft",
    help="radius in feet to snap drainage points to maximum flow accumulation",
    default=0,
    type=float,
)
def basins_cli(
    fdr_file: str,
    fac_file: str | None,
    dp_file: str,
    output_file: str,
    chunk_size: int,
    all_basins: bool,
    dp_layer: str | None,
    snap_radius_ft: float,
):
    """
    Delineate drainage basins from a flow direction raster and drainage points.

    This command labels each cell with the ID of its downstream drainage point,
    effectively delineating basin boundaries.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Watershed delineation", spinner=False):
            with progress_display.progress_context("Label Watersheds"):
                # Load drainage points and FID mapping
                drainage_points, fid_mapping = _drainage_points_from_file(
                    fdr_file, dp_file, dp_layer
                )

                # Snap drainage points to flow accumulation grid if fac_file is provided
                if fac_file is not None and snap_radius_ft > 0:
                    snap_radius_cells = feet_to_cell_count(snap_radius_ft, fdr_file)
                    drainage_points, fid_mapping = snap_drainage_points(
                        drainage_points, fac_file, snap_radius_cells, fid_mapping
                    )

                if chunk_size <= 1:
                    # Non-tiled processing
                    fdr_ds = gdal.Open(fdr_file)
                    if fdr_ds is None:
                        raise ValueError("Could not open flow direction raster file")

                    fdr = fdr_ds.GetRasterBand(1).ReadAsArray()
                    watersheds_arr, graph = label_watersheds(fdr, drainage_points)

                    if not all_basins:
                        # Remove any label not in drainage_points values
                        unique_labels = np.unique(watersheds_arr)
                        for label in unique_labels:
                            if label not in drainage_points.values():
                                watersheds_arr[watersheds_arr == label] = 0

                    # Create output dataset
                    out_ds = create_dataset(
                        output_file,
                        0,
                        gdal.GDT_Int64,
                        fdr.shape[1],
                        fdr.shape[0],
                        fdr_ds.GetGeoTransform(),
                        fdr_ds.GetProjection(),
                    )
                    out_band = out_ds.GetRasterBand(1)
                    out_band.WriteArray(watersheds_arr)
                    out_band.FlushCache()
                    out_ds.FlushCache()
                    out_ds = None
                    out_band = None
                    fdr_ds = None
                else:
                    # Tiled processing
                    graph = _label_watersheds_tiled(
                        fdr_file,
                        drainage_points,
                        output_file,
                        chunk_size,
                        all_basins,
                        progress_display.callback,
                    )

                # Update drainage points file with basin_id and ds_basin_id
                update_drainage_points_file(
                    dp_file, drainage_points, fid_mapping, graph, dp_layer
                )

                resource_stats.add_output_file("Basins", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] basins failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="streams")
@click.option(
    "--fac_file",
    help="path to the GDAL supported raster dataset for the flow accumulation",
    required=True,
)
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the flow direction",
    required=True,
)
@click.option(
    "--output_dir",
    help="path to the output directory",
    required=True,
)
@click.option(
    "--threshold",
    help="minimum flow accumulation threshold (cell count) to define a stream",
    default=5,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 1 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
def streams_cli(
    fac_file: str,
    fdr_file: str,
    output_dir: str,
    threshold: int,
    chunk_size: int,
):
    """
    Extract stream networks from flow accumulation and direction rasters.

    This command identifies stream cells based on a flow accumulation threshold
    and creates vector stream lines and junction points (streams.gpkg).
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Stream extraction", spinner=False):
            with progress_display.progress_context("Extract Streams"):
                streams(
                    fac_file,
                    fdr_file,
                    output_dir,
                    threshold,
                    chunk_size,
                    progress_display.callback,
                )
                resource_stats.add_output_file("Streams", f"{output_dir}/streams.gpkg")
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] streams failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="pipeline")
@click.option(
    "--dem_file",
    help="path to the GDAL supported raster dataset for the DEM",
    required=True,
)
@click.option(
    "--output_dir",
    help="path to the output directory",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size (use <= 0 for in-memory processing)",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--search_radius_ft",
    help="search radius in feet for pit breaching (0 to skip breaching)",
    default=200,
)
@click.option(
    "--max_cost",
    help="maximum cost of breach paths (total sum of elevation removed from each cell in path)",
    default=np.inf,
)
@click.option(
    "--da_sqmi",
    help="minimum drainage area in square miles for stream extraction",
    default=1,
    type=float,
)
@click.option(
    "--basins",
    help="Flag to enable watershed delineation",
    is_flag=True,
)
@click.option(
    "--fill_holes",
    help="If set, fills holes in the DEM",
    is_flag=True,
)
def pipeline_cli(
    dem_file: str,
    output_dir: str,
    chunk_size: int,
    search_radius_ft: float,
    max_cost: float,
    da_sqmi: float,
    basins: bool,
    fill_holes: bool,
):
    """
    Run complete DEM processing pipeline.

    This command runs a full hydrological analysis workflow:
    1. Breach pits (optional, if search_radius_ft > 0)
    2. Fill depressions
    3. Compute flow direction (with flat resolution)
    4. Calculate flow accumulation
    5. Extract streams
    6. Delineate watersheds (optional, if --basins flag is set)
    """
    success = False
    try:
        progress_display = RichProgressDisplay()

        with timer("Total processing", silent=True, spinner=False):
            search_radius = feet_to_cell_count(search_radius_ft, dem_file)
            threshold = sqmi_to_cell_count(da_sqmi, dem_file)

            # Get raster dimensions for breaching when chunk_size <= 0
            ds = gdal.Open(dem_file)
            core_chunk_size = (
                max(ds.RasterXSize, ds.RasterYSize) if chunk_size <= 0 else chunk_size
            )
            ds = None  # Close the dataset

            if search_radius > 0:
                with timer("Breaching", spinner=False):
                    with progress_display.progress_context("Breaching pits"):
                        breach(
                            dem_file,
                            f"{output_dir}/dem_breached.tif",
                            core_chunk_size,
                            search_radius,
                            max_cost,
                            progress_display.callback,
                        )

                with timer("Filling", spinner=False):
                    with progress_display.progress_context("Filling depressions"):
                        fill(
                            f"{output_dir}/dem_breached.tif",
                            f"{output_dir}/dem_filled.tif",
                            chunk_size if chunk_size > 0 else 0,
                            output_dir,
                            fill_holes,
                            progress_display.callback,
                        )
            else:
                with timer("Filling", spinner=False):
                    with progress_display.progress_context("Filling depressions"):
                        fill(
                            dem_file,
                            f"{output_dir}/dem_filled.tif",
                            chunk_size if chunk_size > 0 else 0,
                            output_dir,
                            fill_holes,
                            progress_display.callback,
                        )

            resource_stats.add_output_file(
                "Corrected DEM", f"{output_dir}/dem_filled.tif"
            )

            with timer("Flow direction", spinner=False):
                with progress_display.progress_context("Computing flow direction"):
                    flow_direction(
                        f"{output_dir}/dem_filled.tif",
                        f"{output_dir}/fdr.tif",
                        core_chunk_size,
                        output_dir,
                        resolve_flats=True,
                        progress_callback=progress_display.callback,
                    )

            resource_stats.add_output_file("Flow Direction", f"{output_dir}/fdr.tif")

            with timer("Flow accumulation", spinner=False):
                with progress_display.progress_context("Flow Accumulation"):
                    accumulation(
                        f"{output_dir}/fdr.tif",
                        f"{output_dir}/accum.tif",
                        chunk_size if chunk_size > 0 else 0,
                        progress_display.callback,
                    )

            resource_stats.add_output_file(
                "Flow Accumulation", f"{output_dir}/accum.tif"
            )

            with timer("Stream extraction", spinner=False):
                with progress_display.progress_context("Extract Streams"):
                    streams(
                        f"{output_dir}/accum.tif",
                        f"{output_dir}/fdr.tif",
                        output_dir,
                        threshold,
                        chunk_size if chunk_size > 0 else 0,
                        progress_display.callback,
                    )

            resource_stats.add_output_file("Streams", f"{output_dir}/streams.gpkg")

            if basins:
                with timer("Watershed delineation", spinner=False):
                    with progress_display.progress_context("Label Watersheds"):
                        dp, fid_mapping = _drainage_points_from_file(
                            f"{output_dir}/fdr.tif",
                            f"{output_dir}/streams.gpkg",
                            "junctions",
                        )

                        if chunk_size <= 0:
                            # Non-tiled processing
                            fdr_ds = gdal.Open(f"{output_dir}/fdr.tif")
                            fdr = fdr_ds.GetRasterBand(1).ReadAsArray()
                            watersheds_arr, graph = label_watersheds(fdr, dp)

                            # Create output dataset
                            out_ds = create_dataset(
                                f"{output_dir}/basins.tif",
                                0,
                                gdal.GDT_Int64,
                                fdr.shape[1],
                                fdr.shape[0],
                                fdr_ds.GetGeoTransform(),
                                fdr_ds.GetProjection(),
                            )
                            out_band = out_ds.GetRasterBand(1)
                            out_band.WriteArray(watersheds_arr)
                            out_band.FlushCache()
                            out_ds.FlushCache()
                            out_ds = None
                            out_band = None
                            fdr_ds = None
                        else:
                            graph = _label_watersheds_tiled(
                                f"{output_dir}/fdr.tif",
                                dp,
                                f"{output_dir}/basins.tif",
                                chunk_size,
                                False,
                                progress_display.callback,
                            )

                        # Update drainage points file with basin_id and ds_basin_id
                        update_drainage_points_file(
                            f"{output_dir}/streams.gpkg",
                            dp,
                            fid_mapping,
                            graph,
                            "junctions",
                        )

                resource_stats.add_output_file("Basins", f"{output_dir}/basins.tif")

            success = True

        console.print(resource_stats.get_summary_panel(success=True))

    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] pipeline failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="flow-length")
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the flow direction",
    required=True,
)
@click.option(
    "--dp_file",
    help="path to the drainage points file",
    required=True,
)
@click.option(
    "--output_raster",
    help="path to the output raster file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--output_vector",
    help="path to the output vector file (GeoPackage) for longest flow paths. If not provided, no vector output is created.",
    required=False,
    default=None,
)
@click.option(
    "--fac_file",
    help="path to the flow accumulation raster (optional, for snapping drainage points)",
    required=False,
    default=None,
)
@click.option(
    "--dp_layer",
    help="name of the layer in the drainage points file",
    required=False,
    default=None,
)
@click.option(
    "--snap_radius_ft",
    help="radius in feet to snap drainage points to maximum flow accumulation",
    default=0,
    type=float,
)
def flow_length_cli(
    fdr_file: str,
    dp_file: str,
    output_raster: str,
    output_vector: str | None,
    fac_file: str | None,
    dp_layer: str | None,
    snap_radius_ft: float,
):
    """
    Calculate upstream flow length (longest flow path) from drainage points.

    This command calculates the distance from each cell to its downstream drainage point,
    measured along the flow path. The output raster contains flow length values in map units
    (or meters for geographic CRS).

    The longest flow path for a basin is the cell with the maximum value in that basin.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Calculate flow length", spinner=False):
            with progress_display.progress_context("Calculating flow length"):
                # Calculate snap radius in cells if FAC file is provided
                snap_radius_cells = 0
                if fac_file is not None and snap_radius_ft > 0:
                    snap_radius_cells = feet_to_cell_count(snap_radius_ft, fdr_file)

                _flow_length_core(
                    fdr_file,
                    dp_file,
                    output_raster,
                    output_vector,
                    dp_layer,
                    fac_file,
                    snap_radius_cells,
                )
                resource_stats.add_output_file("Flow Length Raster", output_raster)
                if output_vector is not None:
                    resource_stats.add_output_file("Flow Paths Vector", output_vector)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] flow-length failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


if __name__ == "__main__":
    # run the function

    main()
