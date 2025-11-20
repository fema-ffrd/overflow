import os
import sys

import click
import numpy as np
from osgeo import gdal

from overflow import __version__
from overflow.basins.core import (
    drainage_points_from_file,
    label_watersheds,
    label_watersheds_from_file,
)
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
from overflow.util.cli_progress import RichProgressDisplay
from overflow.util.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_RADIUS
from overflow.util.raster import (
    create_dataset,
    feet_to_cell_count,
    snap_drainage_points,
    sqmi_to_cell_count,
)
from overflow.util.timer import console, resource_stats, timer

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


@main.command(name="breach-single-cell-pits")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
)
@click.option("--output_file", help="path to the output file (must be GeoTiff)")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
def breach_single_cell_pits_cli(input_file: str, output_file: str, chunk_size: int):
    """
    This function is used to breach single cell pits in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with breached single cell pits.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Breach single cell pits", spinner=False):
            with progress_display.progress_context("Breaching single cell pits"):
                breach_single_cell_pits(
                    input_file, output_file, chunk_size, progress_display.callback
                )
                resource_stats.add_output_file("Breached DEM", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="flow-direction")
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
def flow_direction_cli(input_file: str, output_file: str, chunk_size: int):
    """
    This function is used to generate flow direction rasters from chunks of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with delineated flow direction.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Flow direction", spinner=False):
            with progress_display.progress_context("Computing flow direction"):
                flow_direction(
                    input_file, output_file, chunk_size, progress_display.callback
                )
                resource_stats.add_output_file("Flow Direction", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] flow_direction failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="breach-paths-least-cost")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
)
@click.option("--output_file", help="path to the output file (must be GeoTiff)")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
@click.option("--search_radius", help="search radius", default=DEFAULT_SEARCH_RADIUS)
@click.option(
    "--max_cost",
    help="maximum cost of breach paths (total sum of elevation removed from each cell in path)",
    default=np.inf,
)
def breach_paths_least_cost_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
    search_radius: int,
    max_cost: float,
):
    """
    This function is used to breach paths of least cost for pits in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with breached paths of least cost.
    Only pits that can be solved within the search radius are solved.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Breach paths least cost", spinner=False):
            with progress_display.progress_context("Breaching paths (least cost)"):
                breach_paths_least_cost(
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
            f"[bold red]Error:[/bold red] breach_paths_least_cost failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="fix-flats")
@click.option(
    "--dem_file",
    help="path to the GDAL supported raster dataset for the DEM",
    required=True,
)
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the FDR",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--working_dir",
    help="working directory",
)
def fix_flats_cli(
    dem_file: str,
    fdr_file: str,
    output_file: str,
    chunk_size: int,
    working_dir: str | None,
):
    """
    This function is used to fix flats in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with fixed flats.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Fixing flats", spinner=False):
            with progress_display.progress_context("Fixing flats"):
                if chunk_size <= 1:
                    fix_flats_from_file(dem_file, fdr_file, output_file)
                else:
                    fix_flats_tiled(
                        dem_file,
                        fdr_file,
                        output_file,
                        chunk_size,
                        working_dir,
                        progress_display.callback,
                    )
                resource_stats.add_output_file("Fixed FDR", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] fix_flats failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="fill-depressions")
@click.option(
    "--dem_file",
    help="path to the GDAL supported raster dataset for the DEM",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--working_dir",
    help="working directory",
)
@click.option(
    "--fill_holes",
    help="If set, fills holes in the DEM",
    is_flag=True,
)
def fill_depressions_cli(
    dem_file: str,
    output_file: str,
    chunk_size: int,
    working_dir: str | None,
    fill_holes: bool,
):
    """
    This function is used to fill depressions in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with filled depressions.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Filling", spinner=False):
            with progress_display.progress_context("Filling depressions"):
                if chunk_size <= 1:
                    fill_depressions(dem_file, output_file, fill_holes)
                else:
                    fill_depressions_tiled(
                        dem_file,
                        output_file,
                        chunk_size,
                        working_dir,
                        fill_holes,
                        progress_display.callback,
                    )
                resource_stats.add_output_file("Filled DEM", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] fill_depressions failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="flow-accumulation")
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the FDR",
    required=True,
)
@click.option(
    "--output_file",
    help="path to the output file (must be GeoTiff)",
    required=True,
)
@click.option(
    "--chunk_size",
    help="chunk size",
    default=DEFAULT_CHUNK_SIZE,
)
def flow_accumulation_cli(
    fdr_file: str,
    output_file: str,
    chunk_size: int,
):
    """
    This function is used to calculate flow accumulation from a flow direction raster.
    The function takes a flow direction raster as input and returns a flow accumulation raster.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Flow accumulation", spinner=False):
            with progress_display.progress_context("Flow Accumulation"):
                if chunk_size <= 1:
                    flow_accumulation(fdr_file, output_file)
                else:
                    flow_accumulation_tiled(
                        fdr_file, output_file, chunk_size, progress_display.callback
                    )
                resource_stats.add_output_file("Flow Accumulation", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] flow_accumulation failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="label-watersheds")
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the FDR",
    required=True,
)
@click.option(
    "--fac_file",
    help="path to the GDAL supported raster dataset for the flow accumulation (FAC)",
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
    help="chunk size",
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
    default=30,
    type=float,
)
def label_watersheds_cli(
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
    This function is used to label watersheds from a flow direction raster.
    The function takes a flow direction raster and drainage points as input and returns a watersheds raster.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Watershed delineation", spinner=False):
            with progress_display.progress_context("Label Watersheds"):
                # Load drainage points
                drainage_points = drainage_points_from_file(fdr_file, dp_file, dp_layer)

                # Snap drainage points to flow accumulation grid if fac_file is provided
                if fac_file is not None and snap_radius_ft > 0:
                    snap_radius_cells = feet_to_cell_count(snap_radius_ft, fdr_file)
                    drainage_points = snap_drainage_points(
                        drainage_points, fac_file, snap_radius_cells
                    )

                if chunk_size <= 1:
                    # Non-tiled processing
                    fdr_ds = gdal.Open(fdr_file)
                    if fdr_ds is None:
                        raise ValueError("Could not open flow direction raster file")

                    fdr = fdr_ds.GetRasterBand(1).ReadAsArray()
                    watersheds, _ = label_watersheds(fdr, drainage_points)

                    if not all_basins:
                        # Remove any label not in drainage_points values
                        unique_labels = np.unique(watersheds)
                        for label in unique_labels:
                            if label not in drainage_points.values():
                                watersheds[watersheds == label] = 0

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
                    out_band.WriteArray(watersheds)
                    out_band.FlushCache()
                    out_ds.FlushCache()
                    out_ds = None
                    out_band = None
                    fdr_ds = None
                else:
                    # Tiled processing
                    label_watersheds_tiled(
                        fdr_file,
                        drainage_points,
                        output_file,
                        chunk_size,
                        all_basins,
                        progress_display.callback,
                    )
                resource_stats.add_output_file("Watersheds", output_file)
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] label_watersheds failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="extract-streams")
@click.option(
    "--fac_file",
    help="path to the GDAL supported raster dataset for the FAC",
    required=True,
)
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the FDR",
    required=True,
)
@click.option(
    "--output_dir",
    help="path to the output directory",
    required=True,
)
@click.option(
    "--cell_count_threshold",
    help="cell count threshold",
    default=5,
)
@click.option(
    "--chunk_size",
    help="chunk size",
    default=DEFAULT_CHUNK_SIZE,
)
def extract_streams_cli(
    fac_file: str,
    fdr_file: str,
    output_dir: str,
    cell_count_threshold: int,
    chunk_size: int,
):
    """
    This function is used to extract streams from a flow accumulation and flow direction raster.
    The function takes a flow accumulation and flow direction raster as input and returns a streams raster.
    """
    success = False
    try:
        progress_display = RichProgressDisplay()
        with timer("Stream extraction", spinner=False):
            with progress_display.progress_context("Extract Streams"):
                if chunk_size <= 1:
                    extract_streams(
                        fac_file,
                        fdr_file,
                        output_dir,
                        cell_count_threshold,
                        progress_display.callback,
                    )
                else:
                    extract_streams_tiled(
                        fac_file,
                        fdr_file,
                        output_dir,
                        cell_count_threshold,
                        chunk_size,
                        progress_display.callback,
                    )
                resource_stats.add_output_file("Streams", f"{output_dir}/streams.gpkg")
                success = True

        console.print(resource_stats.get_summary_panel(success=success))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] extract_streams failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


@main.command(name="process-dem")
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
    help="chunk size",
    default=DEFAULT_CHUNK_SIZE,
)
@click.option(
    "--search_radius_ft",
    help="search radius in feet to look for solution paths",
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
def process_dem_cli(
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
    This function is used to process a DEM.
    The function takes a DEM as input and returns a streams raster.
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
                    with progress_display.progress_context("Breaching paths"):
                        breach_paths_least_cost(
                            dem_file,
                            f"{output_dir}/dem_corrected.tif",
                            core_chunk_size,
                            search_radius,
                            max_cost,
                            progress_display.callback,
                        )

                with timer("Filling", spinner=False):
                    with progress_display.progress_context("Filling depressions"):
                        if chunk_size <= 0:
                            fill_depressions(
                                f"{output_dir}/dem_corrected.tif", None, fill_holes
                            )
                        else:
                            fill_depressions_tiled(
                                f"{output_dir}/dem_corrected.tif",
                                None,
                                chunk_size,
                                output_dir,
                                fill_holes,
                                progress_display.callback,
                            )
            else:
                with timer("Filling", spinner=False):
                    with progress_display.progress_context("Filling depressions"):
                        if chunk_size <= 0:
                            fill_depressions(
                                dem_file, f"{output_dir}/dem_corrected.tif", fill_holes
                            )
                        else:
                            fill_depressions_tiled(
                                dem_file,
                                f"{output_dir}/dem_corrected.tif",
                                chunk_size,
                                output_dir,
                                fill_holes,
                                progress_display.callback,
                            )

            resource_stats.add_output_file(
                "Corrected DEM", f"{output_dir}/dem_corrected.tif"
            )

            with timer("Flow direction", spinner=False):
                with progress_display.progress_context("Computing flow direction"):
                    flow_direction(
                        f"{output_dir}/dem_corrected.tif",
                        f"{output_dir}/fdr.tif",
                        core_chunk_size,
                        progress_display.callback,
                    )

            resource_stats.add_output_file("Flow Direction", f"{output_dir}/fdr.tif")

            with timer("Fixing flats", spinner=False):
                with progress_display.progress_context("Fixing flats"):
                    if chunk_size <= 0:
                        fix_flats_from_file(
                            f"{output_dir}/dem_corrected.tif",
                            f"{output_dir}/fdr.tif",
                            None,
                        )
                    else:
                        fix_flats_tiled(
                            f"{output_dir}/dem_corrected.tif",
                            f"{output_dir}/fdr.tif",
                            None,
                            chunk_size,
                            output_dir,
                            progress_display.callback,
                        )

            with timer("Flow accumulation", spinner=False):
                with progress_display.progress_context("Flow Accumulation"):
                    if chunk_size <= 0:
                        flow_accumulation(
                            f"{output_dir}/fdr.tif", f"{output_dir}/accum.tif"
                        )
                    else:
                        flow_accumulation_tiled(
                            f"{output_dir}/fdr.tif",
                            f"{output_dir}/accum.tif",
                            chunk_size,
                            progress_display.callback,
                        )

            resource_stats.add_output_file(
                "Flow Accumulation", f"{output_dir}/accum.tif"
            )

            with timer("Stream extraction", spinner=False):
                with progress_display.progress_context("Extract Streams"):
                    if chunk_size <= 0:
                        extract_streams(
                            f"{output_dir}/accum.tif",
                            f"{output_dir}/fdr.tif",
                            output_dir,
                            threshold,
                            progress_display.callback,
                        )
                    else:
                        extract_streams_tiled(
                            f"{output_dir}/accum.tif",
                            f"{output_dir}/fdr.tif",
                            output_dir,
                            threshold,
                            chunk_size,
                            progress_display.callback,
                        )

            resource_stats.add_output_file("Streams", f"{output_dir}/streams.gpkg")

            if basins:
                with timer("Watershed delineation", spinner=False):
                    with progress_display.progress_context("Label Watersheds"):
                        drainage_points = drainage_points_from_file(
                            f"{output_dir}/fdr.tif",
                            f"{output_dir}/streams.gpkg",
                            "junctions",
                        )

                        if chunk_size <= 0:
                            label_watersheds_from_file(
                                f"{output_dir}/fdr.tif",
                                f"{output_dir}/streams.gpkg",
                                f"{output_dir}/basins.tif",
                                False,
                                "junctions",
                            )
                        else:
                            label_watersheds_tiled(
                                f"{output_dir}/fdr.tif",
                                drainage_points,
                                f"{output_dir}/basins.tif",
                                chunk_size,
                                False,
                                progress_display.callback,
                            )

                resource_stats.add_output_file("Basins", f"{output_dir}/basins.tif")

            success = True

        console.print(resource_stats.get_summary_panel(success=True))

    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] process_dem failed with the following exception: {str(exc)}"
        )
        if not success:
            console.print(resource_stats.get_summary_panel(success=False))
        raise click.Abort()


if __name__ == "__main__":
    # run the function

    main()
