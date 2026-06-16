import os

from osgeo import gdal

import overflow
from overflow._util.raster import get_units_to_meters_conversion


def calculate_cell_threshold(area_sqmi, cell_size_ft=4):
    """
    Calculate the number of cells corresponding to a given area in square miles.

    Parameters:
    - area_sqmi: Area in square miles
    - cell_size_ft: Size of each cell in feet (default is 4ft)

    Returns:
    - Number of cells corresponding to the given area
    """
    # Convert area from square miles to square feet
    area_sqft = area_sqmi * (5280**2)  # 1 mile = 5280 feet

    # Calculate the area of each cell in square feet
    cell_area_sqft = cell_size_ft**2

    # Calculate the number of cells corresponding to the given area
    num_cells = area_sqft / cell_area_sqft

    return int(num_cells)


def get_raster_cell_size(raster_path, unit="ft"):
    """
    Determine the cell size of a raster in the specified unit ('ft', 'm', or 'native').

    Parameters:
    - raster_path: Path to the raster file
    - unit: Unit to return the cell size in ('ft' for feet, 'm' for meters, 'native' for native CRS units)

    Returns:
    - Cell size as a float
    """
    dataset = gdal.Open(raster_path)
    if not dataset:
        raise ValueError(f"Could not open raster dataset at {raster_path}")

    gt = dataset.GetGeoTransform()
    cell_size_native = (abs(gt[1]) + abs(gt[5])) / 2.0

    if unit == "native":
        return cell_size_native

    units_to_meters = get_units_to_meters_conversion(dataset)
    cell_size_meters = cell_size_native * units_to_meters

    if unit == "m":
        return cell_size_meters
    elif unit == "ft":
        return cell_size_meters / 0.3048
    else:
        raise ValueError(f"Unknown unit: {unit}. Choose from 'ft', 'm', or 'native'.")


def print_progress(
    phase=None,
    step_name=None,
    step_number=0,
    total_steps=0,
    message="",
    progress=0.0,
):
    parts = []
    if phase:
        parts.append(phase)
    if step_name:
        parts.append(step_name)
    if message:
        parts.append(message)
    if progress > 0:
        parts.append(f"{progress * 100:.1f}%")
    if parts:
        print(" | ".join(parts))


def main():
    # 1. Setup paths and parameters
    chunks = 2048
    dem_file = "/workspaces/overflow/data/input/Allegheny_1m_burned.tif"
    raster_output_dir = "/workspaces/overflow/data/output/rasters"
    if not os.path.exists(raster_output_dir):
        os.makedirs(raster_output_dir)

    # 2. Terrain Conditioning (Breach + Fill)
    # Note: Python API uses cell count
    radius_cells = 50

    overflow.breach(
        dem_file,
        f"{raster_output_dir}/dem_breached.tif",
        search_radius=radius_cells,
        progress_callback=print_progress,
        chunk_size=chunks,
    )
    overflow.fill(
        f"{raster_output_dir}/dem_breached.tif",
        f"{raster_output_dir}/dem_corrected.tif",
        progress_callback=print_progress,
        chunk_size=chunks,
    )

    # 3. Flow Routing
    overflow.flow_direction(
        f"{raster_output_dir}/dem_corrected.tif",
        f"{raster_output_dir}/fdr.tif",
        progress_callback=print_progress,
        chunk_size=chunks,
    )
    overflow.accumulation(
        f"{raster_output_dir}/fdr.tif",
        f"{raster_output_dir}/accum.tif",
        progress_callback=print_progress,
        chunk_size=chunks,
    )

    drainage_area_sqmi = [100, 0.25, 0.0625]
    cell_size_ft = get_raster_cell_size(dem_file, "ft")
    print(f"Detected raster cell size: {cell_size_ft:.4f} ft")

    for da in drainage_area_sqmi:
        threshold_cells = calculate_cell_threshold(da, cell_size_ft)
        print(f"Using a threshold of {threshold_cells} cells for stream extraction.")

        vector_output_dir = (
            f"/workspaces/overflow/data/output/streams_threshold_{threshold_cells}"
        )
        if not os.path.exists(vector_output_dir):
            os.makedirs(vector_output_dir)

        # 4. Feature Extraction
        overflow.streams(
            fac_path=f"{raster_output_dir}/accum.tif",
            fdr_path=f"{raster_output_dir}/fdr.tif",
            output_dir=vector_output_dir,
            threshold=threshold_cells,  # minimum number of upstream cells required to be considered a stream.
            progress_callback=print_progress,
        )

        # 5. Basins
        # Uses the junctions layer from the generated streams.gpkg
        overflow.basins(
            fdr_path=f"{raster_output_dir}/fdr.tif",
            drainage_points_path=f"{vector_output_dir}/streams.gpkg",
            output_path=f"{vector_output_dir}/basins.tif",
            layer_name="junctions",
            progress_callback=print_progress,
        )


if __name__ == "__main__":
    main()
