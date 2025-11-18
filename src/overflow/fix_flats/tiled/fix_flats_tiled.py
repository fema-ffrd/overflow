import os
import shutil
import tempfile

from osgeo import gdal
from rich.console import Console

from overflow.fix_flats.tiled.flat_mask import create_flat_mask
from overflow.fix_flats.tiled.global_state import create_global_state
from overflow.fix_flats.tiled.update_fdr import update_fdr
from overflow.util.constants import DEFAULT_CHUNK_SIZE, FLOW_DIRECTION_NODATA
from overflow.util.raster import create_dataset, open_dataset

LABELS_FILENAME = "labels.tif"
FLAT_MASK_FILENAME = "flat_mask.tif"


def setup_datasets(
    dem_filepath: str,
    fdr_filepath: str,
    fixed_fdr_filepath: str | None,
    working_dir: str,
) -> tuple[gdal.Dataset, gdal.Dataset, gdal.Dataset, gdal.Dataset, gdal.Dataset]:
    """
    Set up the necessary datasets for the fix_flats_tiled function.

    This function opens the input DEM and flow direction datasets, creates the output datasets
    for the fixed flow direction, labels, and flat mask, and returns the dataset objects.

    Args:
        dem_filepath (str): The file path of the input DEM dataset.
        fdr_filepath (str): The file path of the input flow direction dataset.
        fixed_fdr_filepath (str): The file path of the output fixed flow direction dataset.
        working_dir (str): The directory where the temporary datasets will be created.

    Returns:
        tuple: A tuple containing the following dataset objects:
            - dem_ds (gdal.Dataset): The input DEM dataset.
            - fdr_ds (gdal.Dataset): The input flow direction dataset.
            - fixed_fdr_ds (gdal.Dataset): The output fixed flow direction dataset.
            - labels_ds (gdal.Dataset): The temporary labels dataset.
            - flat_mask_ds (gdal.Dataset): The temporary flat mask dataset.
    """
    dem_ds = open_dataset(dem_filepath)
    fdr_ds = open_dataset(
        fdr_filepath, gdal.GA_Update if fixed_fdr_filepath is None else gdal.GA_ReadOnly
    )
    dem_band = dem_ds.GetRasterBand(1)
    x_size = dem_band.XSize
    y_size = dem_band.YSize
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    if fixed_fdr_filepath is None:
        fixed_fdr_ds = fdr_ds
    else:
        fixed_fdr_ds = create_dataset(
            fixed_fdr_filepath,
            FLOW_DIRECTION_NODATA,
            gdal.GDT_Byte,
            x_size,
            y_size,
            geotransform,
            projection,
        )
    labels_ds = create_dataset(
        os.path.join(working_dir, LABELS_FILENAME),
        0,
        gdal.GDT_Int64,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    flat_mask_ds = create_dataset(
        os.path.join(working_dir, FLAT_MASK_FILENAME),
        0,
        gdal.GDT_Int64,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    return (dem_ds, fdr_ds, fixed_fdr_ds, labels_ds, flat_mask_ds)


def fix_flats_tiled(
    dem_filepath: str,
    fdr_filepath: str,
    output_filepath: str | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str | None = None,
) -> None:
    """
    Fix flats in a DEM using a tiled approach.

    This function fixes flats in a DEM by processing the DEM in chunks (tiles) and updating the
    flow direction raster. It creates a global state object to manage the connectivity between
    tiles, computes the distances to low and high edge tiles, creates a flat mask, and updates
    the flow direction raster.

    Args:
        dem_filepath (str): The file path of the input DEM dataset.
        fdr_filepath (str): The file path of the input flow direction dataset.
        output_filepath (str): The file path of the output fixed flow direction dataset.
        chunk_size (int, optional): The size of each chunk (tile) to process the DEM.
            Default is DEFAULT_CHUNK_SIZE.
        working_dir (str | None, optional): The directory where the temporary datasets will be created.
            If not provided, a temporary directory will be created and cleaned up after processing.

    Returns:
        None

    Notes:
        - The fix_flats_tiled function implements a parallel algorithm for fixing flats in a DEM.
        - The parallel algorithm is expected to be slower than the sequential algorithm due to
          several factors:
          - Unbalanced workload distribution among tiles, leading to some consumer processes
            having significantly more work than others.
          - Overhead of constructing local HighGraph and LowGraph for tiles with flat cells on
            borders, which involves time-consuming region-growing procedures.
          - Communication and synchronization overhead between consumer processes.
          - Limited parallelization opportunities within each tile.
        - Despite the slower performance, the parallel algorithm scales with the number of
          processors, allowing for the processing of larger DEMs in a reasonable amount of time.
        - The parallel algorithm operates with significantly less RAM compared to the sequential
          algorithm, enabling the processing of DEMs that may not be possible with the sequential
          algorithm due to memory constraints.
        - The scalability and memory efficiency of the parallel algorithm come with the trade-off
          of potentially slower execution times for smaller DEMs or DEMs with larger flat regions.
    """
    console = Console()
    cleanup_working_dir = False
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
        cleanup_working_dir = True
    dem_ds, fdr_ds, fixed_fdr_ds, labels_ds, flat_mask_ds = setup_datasets(
        dem_filepath, fdr_filepath, output_filepath, working_dir
    )
    dem_band = dem_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    fixed_fdr_band = fixed_fdr_ds.GetRasterBand(1)
    labels_band = labels_ds.GetRasterBand(1)
    flat_mask_band = flat_mask_ds.GetRasterBand(1)

    print("Step 1 of 3: Creating global state")

    global_state = create_global_state(dem_band, fdr_band, labels_band, chunk_size)

    with console.status("Solving graph..."):
        dist_to_low_edge_tiles, dist_to_high_edge_tiles = (
            global_state.graph.solve_graph()
        )

    print("Step 2 of 3: Creating flat mask")

    create_flat_mask(
        chunk_size,
        flat_mask_band,
        labels_band,
        fdr_band,
        global_state,
        dist_to_high_edge_tiles,
        dist_to_low_edge_tiles,
    )

    print("Step 3 of 3: Updating flow direction")

    update_fdr(dem_band, fdr_band, fixed_fdr_band, flat_mask_band, chunk_size)
    if cleanup_working_dir:
        shutil.rmtree(working_dir)
    dem_ds = None
    fdr_ds = None
    fixed_fdr_ds = None
    labels_ds = None
    flat_mask_ds = None
