#!/usr/bin/env python3
"""
Generate visualization images for overflow operations.

Creates before/after images showing how each algorithm transforms the input data.
"""

import os
import tempfile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from osgeo import gdal, ogr, osr

import overflow


def create_terrain_colormap():
    colors = [
        "#8B7D6B",
        "#A0826D",
        "#B8956E",
        "#D4AF6A",
        "#E5D352",
        "#C8DC5A",
        "#9BC872",
        "#7AB87D",
        "#5AAA88",
        "#3D9C93",
        "#1E8E9E",
        "#0080A9",
        "#0072B4",
        "#0064BF",
    ]
    return LinearSegmentedColormap.from_list("terrain", colors, N=100)


def create_geotiff(data, output_path, nodata_value=-9999.0):
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(float(nodata_value))
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def read_geotiff(input_path):
    dataset = gdal.Open(input_path)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    dataset = None
    return data


def plot_dem_grid(
    dem,
    highlighted_cells=None,
    title="",
    output_path=None,
    cmap=None,
    vmin=None,
    vmax=None,
    value_format="auto",
):
    rows, cols = dem.shape
    _, ax = plt.subplots(figsize=(10, 10))
    if cmap is None:
        cmap = create_terrain_colormap()
    if vmin is None:
        vmin = np.nanmin(dem)
    if vmax is None:
        vmax = np.nanmax(dem)

    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    flow_arrows = {
        0: "→",  # EAST
        1: "↗",  # NORTH_EAST
        2: "↑",  # NORTH
        3: "↖",  # NORTH_WEST
        4: "←",  # WEST
        5: "↙",  # SOUTH_WEST
        6: "↓",  # SOUTH
        7: "↘",  # SOUTH_EAST
        8: "○",  # UNDEFINED
        9: "ND",  # NODATA
    }

    for r in range(rows):
        for c in range(cols):
            value = dem[r, c]

            if np.isnan(value):
                text = "ND"
            elif value_format == "arrow":
                text = flow_arrows.get(int(value), "?")
            elif value_format == "int":
                text = f"{int(value)}"
            elif value_format == "float":
                text = f"{value:.1f}"
            else:
                text = f"{int(value)}" if value == int(value) else f"{value:.1f}"

            fontsize = 20 if value_format == "arrow" else 14
            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="black",
            )

    if highlighted_cells:
        for r, c in highlighted_cells:
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                linewidth=6,
                edgecolor="#FF0000",
                facecolor="none",
                zorder=10,
            )
            ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_flow_direction_overlay(dem, flow_dir, output_path):
    rows, cols = dem.shape
    _, ax = plt.subplots(figsize=(10, 10))
    terrain_cmap = create_terrain_colormap()
    ax.imshow(
        dem,
        cmap=terrain_cmap,
        interpolation="nearest",
        vmin=np.nanmin(dem),
        vmax=np.nanmax(dem),
    )

    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    flow_arrows = {
        0: "→",  # EAST
        1: "↗",  # NORTH_EAST
        2: "↑",  # NORTH
        3: "↖",  # NORTH_WEST
        4: "←",  # WEST
        5: "↙",  # SOUTH_WEST
        6: "↓",  # SOUTH
        7: "↘",  # SOUTH_EAST
        8: "○",  # UNDEFINED
        9: "ND",  # NODATA
    }

    for r in range(rows):
        for c in range(cols):
            value = flow_dir[r, c]
            if not np.isnan(value):
                arrow = flow_arrows.get(int(value), "?")
                ax.text(
                    c,
                    r,
                    arrow,
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color="black",
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def create_breach_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            dem[r, c] = 108 - (r + c)

    # Pit
    dem[3, 3] = 95.0

    # Ridge surrounding pit
    dem[4, 4] = 110.0
    dem[4, 5] = 108.0
    dem[5, 4] = 108.0

    # Lower ridge cells to create breach path
    dem[3, 4] = 99.0
    dem[3, 5] = 98.0
    dem[3, 6] = 97.0
    dem[4, 6] = 96.0
    dem[5, 6] = 95.0

    # Path cells that breach will modify
    breach_path = [(3, 3), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6)]

    # Save input DEM
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "breach_input.tif")
        output_path = os.path.join(tmpdir, "breach_output.tif")
        create_geotiff(dem, input_path)
        overflow.breach(
            input_path=input_path, output_path=output_path, search_radius=50
        )
        dem_breached = read_geotiff(output_path)

    breach_dir = os.path.join(output_dir, "breach")
    os.makedirs(breach_dir, exist_ok=True)
    plot_dem_grid(
        dem,
        highlighted_cells=breach_path,
        output_path=os.path.join(breach_dir, "before.png"),
    )
    plot_dem_grid(
        dem_breached,
        highlighted_cells=breach_path,
        output_path=os.path.join(breach_dir, "after.png"),
    )


def create_fill_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            dem[r, c] = 108 - (r + c)

    # Create a larger bowl-shaped depression in the center
    depression_cells = [
        (3, 3),
        (3, 4),
        (3, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (5, 3),
        (5, 4),
        (5, 5),
        (6, 4),
        (6, 5),
    ]

    # Create bowl shape
    dem[4, 4] = 92.0
    dem[3, 4] = 93.0
    dem[4, 3] = 93.0
    dem[4, 5] = 93.5
    dem[5, 4] = 93.5
    dem[3, 3] = 94.0
    dem[3, 5] = 94.5
    dem[5, 3] = 94.5
    dem[5, 5] = 95.0
    dem[6, 4] = 95.5
    dem[6, 5] = 96.0

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "fill_input.tif")
        output_path = os.path.join(tmpdir, "fill_output.tif")
        create_geotiff(dem, input_path)
        overflow.fill(input_path=input_path, output_path=output_path)
        dem_filled = read_geotiff(output_path)

    fill_dir = os.path.join(output_dir, "fill")
    os.makedirs(fill_dir, exist_ok=True)
    plot_dem_grid(
        dem,
        highlighted_cells=depression_cells,
        output_path=os.path.join(fill_dir, "before.png"),
    )
    plot_dem_grid(
        dem_filled,
        highlighted_cells=depression_cells,
        output_path=os.path.join(fill_dir, "after.png"),
    )


def create_flow_direction_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            distance_from_center = abs(c - cols // 2)
            dem[r, c] = 100 - r + distance_from_center * 2

    dem[1:4, 6:9] = 102.0  # Flat plateau

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "flowdir_input.tif")
        output_path = os.path.join(tmpdir, "flowdir_output.tif")
        create_geotiff(dem, input_path)
        overflow.flow_direction(
            input_path=input_path, output_path=output_path, resolve_flats=True
        )
        flow_dir = read_geotiff(output_path)

    flowdir_dir = os.path.join(output_dir, "flow-direction")
    os.makedirs(flowdir_dir, exist_ok=True)
    plot_dem_grid(dem, output_path=os.path.join(flowdir_dir, "input.png"))
    plot_flow_direction_overlay(
        dem, flow_dir, output_path=os.path.join(flowdir_dir, "output.png")
    )


def create_flow_accumulation_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            distance_from_center = abs(c - cols // 2)
            dem[r, c] = 100 - r + distance_from_center * 2

    dem[1:4, 6:9] = 102.0

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path = os.path.join(tmpdir, "dem.tif")
        filled_path = os.path.join(tmpdir, "filled.tif")
        flowdir_path = os.path.join(tmpdir, "flowdir.tif")
        flowacc_path = os.path.join(tmpdir, "flowacc.tif")
        create_geotiff(dem, dem_path)
        overflow.fill(input_path=dem_path, output_path=filled_path)
        overflow.flow_direction(
            input_path=filled_path, output_path=flowdir_path, resolve_flats=True
        )
        overflow.accumulation(input_path=flowdir_path, output_path=flowacc_path)
        flow_dir = read_geotiff(flowdir_path)
        flow_acc = read_geotiff(flowacc_path)

    flowacc_dir = os.path.join(output_dir, "flow-accumulation")
    os.makedirs(flowacc_dir, exist_ok=True)
    plot_flow_direction_overlay(
        dem, flow_dir, output_path=os.path.join(flowacc_dir, "input.png")
    )
    acc_cmap = plt.cm.Blues
    plot_dem_grid(
        flow_acc,
        output_path=os.path.join(flowacc_dir, "output.png"),
        cmap=acc_cmap,
        value_format="int",
    )


def create_streams_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            distance_from_center = abs(c - cols // 2)
            dem[r, c] = 100 - r + distance_from_center * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path = os.path.join(tmpdir, "dem.tif")
        filled_path = os.path.join(tmpdir, "filled.tif")
        flowdir_path = os.path.join(tmpdir, "flowdir.tif")
        flowacc_path = os.path.join(tmpdir, "flowacc.tif")
        create_geotiff(dem, dem_path)
        overflow.fill(input_path=dem_path, output_path=filled_path)
        overflow.flow_direction(
            input_path=filled_path, output_path=flowdir_path, resolve_flats=True
        )
        overflow.accumulation(input_path=flowdir_path, output_path=flowacc_path)
        flow_acc = read_geotiff(flowacc_path)

    threshold = 8
    streams = (flow_acc >= threshold).astype(np.float32)

    streams_dir = os.path.join(output_dir, "streams")
    os.makedirs(streams_dir, exist_ok=True)
    plot_dem_grid(
        flow_acc,
        output_path=os.path.join(streams_dir, "input.png"),
        cmap=plt.cm.Blues,
        value_format="int",
    )
    streams_cmap = ListedColormap(["#E8E8E8", "#0066CC"])
    plot_dem_grid(
        streams,
        output_path=os.path.join(streams_dir, "output.png"),
        cmap=streams_cmap,
        vmin=0,
        vmax=1,
        value_format="int",
    )


def create_basins_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            # Two valleys at columns 2 and 7
            dist_left = abs(c - 2)
            dist_right = abs(c - 7)
            dist = min(dist_left, dist_right)
            dem[r, c] = 100 - r + dist * 3

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path = os.path.join(tmpdir, "dem.tif")
        filled_path = os.path.join(tmpdir, "filled.tif")
        flowdir_path = os.path.join(tmpdir, "flowdir.tif")
        flowacc_path = os.path.join(tmpdir, "flowacc.tif")
        drainage_points_path = os.path.join(tmpdir, "drainage_points.gpkg")
        basins_path = os.path.join(tmpdir, "basins.tif")

        create_geotiff(dem, dem_path)
        overflow.fill(input_path=dem_path, output_path=filled_path)
        overflow.flow_direction(
            input_path=filled_path, output_path=flowdir_path, resolve_flats=True
        )
        overflow.accumulation(input_path=flowdir_path, output_path=flowacc_path)

        # Create drainage points (outlets at bottom of each valley)
        driver = ogr.GetDriverByName("GPKG")
        ds = driver.CreateDataSource(drainage_points_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer("drainage_points", srs, ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

        # Left valley outlet at column 2, bottom row
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("id", 1)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(2.5, -9.5)  # X=col+0.5, Y=-(row+0.5)
        feat.SetGeometry(point)
        layer.CreateFeature(feat)

        # Right valley outlet at column 7, bottom row
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("id", 2)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(7.5, -9.5)
        feat.SetGeometry(point)
        layer.CreateFeature(feat)

        ds = None

        # Run basins operation
        overflow.basins(
            fdr_path=flowdir_path,
            drainage_points_path=drainage_points_path,
            output_path=basins_path,
            all_basins=True,
        )

        flow_acc = read_geotiff(flowacc_path)
        basins_raw = read_geotiff(basins_path)

    # Remap basin IDs to simple sequential values (1, 2)
    basins = np.zeros_like(basins_raw)
    unique_ids = np.unique(basins_raw[basins_raw > 0])
    for i, basin_id in enumerate(sorted(unique_ids), start=1):
        basins[basins_raw == basin_id] = i

    basins_dir = os.path.join(output_dir, "basins")
    os.makedirs(basins_dir, exist_ok=True)
    plot_dem_grid(
        flow_acc,
        output_path=os.path.join(basins_dir, "input.png"),
        cmap=plt.cm.Blues,
        value_format="int",
    )
    basins_cmap = ListedColormap(["#FFB366", "#66B3FF"])  # Orange and blue
    plot_dem_grid(
        basins,
        output_path=os.path.join(basins_dir, "output.png"),
        cmap=basins_cmap,
        vmin=1,
        vmax=2,
        value_format="int",
    )


def create_flow_length_example(output_dir):
    rows, cols = 10, 10
    dem = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            dem[r, c] = 108 - (r + c)

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path = os.path.join(tmpdir, "dem.tif")
        filled_path = os.path.join(tmpdir, "filled.tif")
        flowdir_path = os.path.join(tmpdir, "flowdir.tif")
        drainage_points_path = os.path.join(tmpdir, "drainage_points.gpkg")
        flowlen_path = os.path.join(tmpdir, "flowlen.tif")

        create_geotiff(dem, dem_path)
        overflow.fill(input_path=dem_path, output_path=filled_path)
        overflow.flow_direction(
            input_path=filled_path, output_path=flowdir_path, resolve_flats=True
        )

        # Create drainage point at outlet (lower-right corner)
        driver = ogr.GetDriverByName("GPKG")
        ds = driver.CreateDataSource(drainage_points_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer("drainage_points", srs, ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

        # Outlet at lower-right
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("id", 1)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(9.5, -9.5)  # X=col+0.5, Y=-(row+0.5)
        feat.SetGeometry(point)
        layer.CreateFeature(feat)

        ds = None

        # Run flow_length operation
        overflow.flow_length(
            fdr_path=flowdir_path,
            drainage_points_path=drainage_points_path,
            output_raster=flowlen_path,
        )

        flow_len = read_geotiff(flowlen_path)

    flowlen_dir = os.path.join(output_dir, "flow-length")
    os.makedirs(flowlen_dir, exist_ok=True)
    plot_dem_grid(dem, output_path=os.path.join(flowlen_dir, "input.png"))
    flowlen_cmap = plt.cm.YlOrRd
    plot_dem_grid(
        flow_len,
        output_path=os.path.join(flowlen_dir, "output.png"),
        cmap=flowlen_cmap,
        value_format="float",
    )


def main():
    output_dir = "docs/img"
    os.makedirs(output_dir, exist_ok=True)

    create_breach_example(output_dir)
    create_fill_example(output_dir)
    create_flow_direction_example(output_dir)
    create_flow_accumulation_example(output_dir)
    create_streams_example(output_dir)
    create_basins_example(output_dir)
    create_flow_length_example(output_dir)


if __name__ == "__main__":
    main()
