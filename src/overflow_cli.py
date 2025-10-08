import os
import subprocess
import click
import numpy as np
from osgeo import gdal

from overflow.breach_single_cell_pits import breach_single_cell_pits
from overflow.flow_direction import flow_direction
from overflow.breach_paths_least_cost import (
    breach_paths_least_cost,
    breach_paths_least_cost_cuda,
)
from overflow.fix_flats.tiled import fix_flats_tiled
from overflow.fix_flats.core import fix_flats_from_file
from overflow.fill_depressions.core import fill_depressions
from overflow.fill_depressions.tiled import fill_depressions_tiled
from overflow.flow_accumulation.core import flow_accumulation
from overflow.flow_accumulation.tiled import flow_accumulation_tiled
from overflow.basins.core import label_watersheds_from_file, drainage_points_from_file
from overflow.basins.tiled import label_watersheds_tiled
from overflow.extract_streams.core import extract_streams
from overflow.extract_streams.tiled import extract_streams_tiled
from overflow.util.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_RADIUS
from overflow.util.raster import sqmi_to_cell_count, feet_to_cell_count
from overflow.util.timer import timer, console, resource_stats


# set gdal configuration
gdal.UseExceptions()
gdal.SetConfigOption("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", ""))
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))
gdal.SetConfigOption("CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE", "YES")


@click.group()
def main():
    """The main entry point for the command line interface."""


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

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing

    Returns
    -------
    None
    """
    try:

        breach_single_cell_pits(input_file, output_file, chunk_size)
    except Exception as exc:
        print(
            f"breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies


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

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing

    Returns
    -------
    None
    """
    try:
        flow_direction(input_file, output_file, chunk_size)
    except Exception as exc:
        print(f"flow_direction failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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
@click.option(
    "--cuda",
    help="use experimental CUDA implementation",
    is_flag=True,
)
@click.option(
    "--max_pits",
    help="maximum number of pits to process in each chunk for cuda implementation",
    default=10000,
)
def breach_paths_least_cost_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
    search_radius: int,
    max_cost: float,
    cuda: bool,
    max_pits: int,
):
    """
    This function is used to breach paths of least cost for pits in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with breached paths of least cost.
    Only pits that can be solved within the search radius are solved.

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
    search_radius : int
        Search radius in cells to look for solution paths. Larger search radius will use more memory.

    Returns
    -------
    None
    """
    try:
        if cuda:
            breach_paths_least_cost_cuda(
                input_file, output_file, chunk_size, search_radius, max_pits, max_cost
            )
        else:
            breach_paths_least_cost(
                input_file, output_file, chunk_size, search_radius, max_cost
            )
    except Exception as exc:
        print(
            f"breach_paths_least_cost failed with the following exception: {str(exc)}"
        )
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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

    Parameters
    ----------
    dem_file : str
        Path to the input dem file
    fdr_file : str
        Path to the input flow direction file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing

    Returns
    -------
    None
    """
    try:
        if chunk_size <= 1:
            fix_flats_from_file(dem_file, fdr_file, output_file)
        else:
            fix_flats_tiled(dem_file, fdr_file, output_file, chunk_size, working_dir)
    except Exception as exc:
        print(f"fix_flats failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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

    Parameters
    ----------
    dem_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
        If chunk_size is less than or equal to 1, the fill_depressions function is called
        which fills depressions with an all in RAM algorithm.
    working_dir : str
        Working directory to store temporary files
    fill_holes : bool
        If set, fills holes in the DEM

    Returns
    -------
    None
    """
    try:
        if chunk_size <= 1:
            fill_depressions(dem_file, output_file, fill_holes)
        else:
            fill_depressions_tiled(
                dem_file, output_file, chunk_size, working_dir, fill_holes
            )
    except Exception as exc:
        print(f"fill_depressions failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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

    Parameters
    ----------
    fdr_file : str
        Path to the input flow direction file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
        If chunk_size is less than or equal to 1, the flow_accumulation function is called
        which calculates flow accumulation with an all in RAM algorithm.

    Returns
    -------
    None
    """
    try:
        if chunk_size <= 1:
            flow_accumulation(fdr_file, output_file)
        else:
            flow_accumulation_tiled(fdr_file, output_file, chunk_size)
    except Exception as exc:
        print(f"flow_accumulation failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
        raise click.Abort()


@main.command(name="label-watersheds")
@click.option(
    "--fdr_file",
    help="path to the GDAL supported raster dataset for the FDR",
    required=True,
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
def label_watersheds_cli(
    fdr_file: str,
    dp_file: str,
    output_file: str,
    chunk_size: int,
    all_basins: bool,
    dp_layer: str | None,
):
    """
    This function is used to label watersheds from a flow direction raster.
    The function takes a flow direction raster and drainage points as input and returns a watersheds raster.

    Parameters
    ----------
    fdr_file : str
        Path to the input flow direction file
    dp_file : str
        Path to the drainage points file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
        If chunk_size is less than or equal to 1, the label_watersheds function is called
        which calculates watersheds with an all in RAM algorithm.

    Returns
    -------
    None
    """
    try:
        # TODO, snap the drainage points to the flow accumulation grid
        if chunk_size <= 1:
            label_watersheds_from_file(
                fdr_file, dp_file, output_file, all_basins, dp_layer
            )
        else:
            drainage_points = drainage_points_from_file(fdr_file, dp_file, dp_layer)
            label_watersheds_tiled(
                fdr_file, drainage_points, output_file, chunk_size, all_basins
            )
    except Exception as exc:
        print(f"label_watersheds failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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

    Parameters
    ----------
    fac_file : str
        Path to the input flow accumulation file
    fdr_file : str
        Path to the input flow direction file
    output_dir : str
        Path to the output directory
    cell_count_threshold : int
        Cell count threshold
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
        If chunk_size is less than or equal to 1, the extract_streams function is called
        which extracts streams with an all in RAM algorithm.

    Returns
    -------
    None
    """
    try:
        if chunk_size <= 1:
            extract_streams(fac_file, fdr_file, output_dir, cell_count_threshold)
        else:
            extract_streams_tiled(
                fac_file, fdr_file, output_dir, cell_count_threshold, chunk_size
            )
    except Exception as exc:
        print(f"extract_streams failed with the following exception: {str(exc)}")
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
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

    Parameters
    ----------
    dem_file : str
        Path to the input dem file
    output_dir : str
        Path to the output directory
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
        If chunk_size is 0 or negative, core algorithms will be used instead of tiled processing
        (except for breaching which will use the larger of the raster dimensions as chunk size).
    search_radius_ft : float
        Search radius in feet to look for solution paths. Larger search radius will use more memory.
    max_cost : float
        Maximum cost of breach paths (total sum of elevation removed from each cell in path)
    da_sqmi : float
        Minimum drainage area in square miles for stream extraction
    basins : bool
        Flag to enable watershed delineation
    fill_holes : bool
        If set, fills holes in the DEM

    Returns
    -------
    None
    """
    try:
        console.rule("[bold blue]DEM Processing")

        with timer("Total processing", silent=True, spinner=False):
            search_radius = feet_to_cell_count(search_radius_ft, dem_file)
            threshold = sqmi_to_cell_count(da_sqmi, dem_file)

            # Get raster dimensions for breaching when chunk_size <= 0
            ds = gdal.Open(dem_file)
            core_chunk_size = (
                max(ds.RasterXSize, ds.RasterYSize) if chunk_size <= 0 else chunk_size
            )
            ds = None  # Close the dataset

            # Print configuration
            console.print("\n[yellow]Configuration:[/yellow]")
            console.print(f"  DEM file: [cyan]{dem_file}[/cyan]")
            console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
            console.print(
                f"  Processing mode: [cyan]{'in-memory' if chunk_size <= 0 else 'tiled'} (chunk size: {core_chunk_size if chunk_size <= 0 else chunk_size})[/cyan]"
            )
            console.print(
                f"  Breach search radius: [cyan]{search_radius_ft}ft ({search_radius} cells)[/cyan]"
            )
            console.print(f"  Maximum cost: [cyan]{max_cost}[/cyan]")
            console.print(
                f"  Drainage area threshold: [cyan]{da_sqmi} sqmi ({threshold} cells)[/cyan]"
            )
            console.print(f"  Extract basins: [cyan]{basins}[/cyan]")
            console.print(f"  Fill holes: [cyan]{fill_holes}[/cyan]")

            if search_radius > 0:
                with timer("Breaching", spinner=False):
                    breach_paths_least_cost(
                        dem_file,
                        f"{output_dir}/dem_corrected.tif",
                        core_chunk_size,
                        search_radius,
                        max_cost,
                    )

                with timer("Filling", spinner=False):
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
                        )
            else:
                with timer("Filling", spinner=False):
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
                        )

            with timer("Flow direction", spinner=False):
                flow_direction(
                    f"{output_dir}/dem_corrected.tif",
                    f"{output_dir}/fdr.tif",
                    core_chunk_size,
                )

            with timer("Fixing flats", spinner=False):
                if chunk_size <= 0:
                    fix_flats_from_file(
                        f"{output_dir}/dem_corrected.tif", f"{output_dir}/fdr.tif", None
                    )
                else:
                    fix_flats_tiled(
                        f"{output_dir}/dem_corrected.tif",
                        f"{output_dir}/fdr.tif",
                        None,
                        chunk_size,
                        output_dir,
                    )

            with timer("Flow accumulation", spinner=False):
                if chunk_size <= 0:
                    flow_accumulation(
                        f"{output_dir}/fdr.tif", f"{output_dir}/accum.tif"
                    )
                else:
                    flow_accumulation_tiled(
                        f"{output_dir}/fdr.tif", f"{output_dir}/accum.tif", chunk_size
                    )

            with timer("Stream extraction", spinner=False):
                if chunk_size <= 0:
                    extract_streams(
                        f"{output_dir}/accum.tif",
                        f"{output_dir}/fdr.tif",
                        output_dir,
                        threshold,
                    )
                else:
                    extract_streams_tiled(
                        f"{output_dir}/accum.tif",
                        f"{output_dir}/fdr.tif",
                        output_dir,
                        threshold,
                        chunk_size,
                    )

            if basins:
                with timer("Watershed delineation", spinner=False):
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
                        )

        console.rule("[bold green]Processing Complete")

        # Display resource summary chart
        console.print(resource_stats.get_chart())

    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] process_dem failed with the following exception: {str(exc)}"
        )
        raise click.Abort()


@main.command(name="test")
@click.option(
    "--all",
    "run_all",
    help="Run all tests including CUDA tests",
    is_flag=True,
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def test_cli(run_all: bool, pytest_args: tuple):
    """
    Run pytest tests. By default, excludes CUDA tests with -k "not cuda".
    Use --all to run all tests including CUDA tests.
    Additional pytest arguments can be passed after the command options.

    Examples:
        overflow test                    # Run tests excluding CUDA
        overflow test --all              # Run all tests including CUDA
        overflow test tests/             # Run specific test directory
        overflow test -v                 # Run with verbose output
    """
    try:
        cmd = ["pytest"]

        # Add -k "not cuda" flag by default unless --all is specified
        if not run_all:
            cmd.extend(["-k", "not cuda"])

        # Add any additional pytest arguments
        if pytest_args:
            cmd.extend(pytest_args)

        # Run pytest
        result = subprocess.run(cmd, cwd=os.getcwd())

        # Exit with the same code as pytest
        raise SystemExit(result.returncode)

    except FileNotFoundError:
        print("Error: pytest not found. Make sure pytest is installed.")
        raise click.Abort()
    except Exception as exc:
        print(f"test command failed with the following exception: {str(exc)}")
        raise click.Abort()


if __name__ == "__main__":
    # run the function
    # pylint does not understand click decorators
    # pylint: disable=no-value-for-parameter
    main()
