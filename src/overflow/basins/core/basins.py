from osgeo import gdal, ogr, osr
import numpy as np
from numba import njit, prange, int64
from numba.typed import Dict  # pylint: disable=no-name-in-module
from numba.types import UniTuple
from overflow.util.constants import (
    NEIGHBOR_OFFSETS,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_NODATA,
)
from overflow.util.queue import GridCellInt64Queue as Queue
from overflow.util.raster import GridCellInt64 as GridCell
from overflow.util.raster import create_dataset
from .basin_polygons import create_basin_polygons

gdal.UseExceptions()


def drainage_points_from_file(
    fdr_filepath: str, drainage_points_file: str, layer_name: None | str = None
) -> dict:
    """
    Read drainage points from an OGR compatible file and return a dictionary for use in the label_watersheds function.

    Parameters:
    - fdr_filepath (str): The path to the flow direction raster file.
                          The flow direction raster is used to convert drainage point coordinates to row and column
                          indices.
    - drainage_points_file (str): The path to the drainage points file.
                                    The file should be in an OGR compatible format (e.g., Shapefile, GeoPackage).
    - layer_name (str): The name of the layer in the drainage points file to read.
                        If None, the first layer in the file will be used. Default is None.

    Returns:
    - dict: A dictionary containing the drainage points.
            The keys are tuples (row, col) representing the coordinates of the drainage points,
            and the values are the corresponding watershed IDs that will be populated by the label_watersheds function.

    Description:
    The function reads the drainage points from an OGR compatible file using the OGR library.
    It opens the file and reads the features from the layer.
    For each feature, it retrieves the geometry and extracts the coordinates.
    The coordinates are converted to row and column indices, and the watershed ID is extracted from the feature.
    The function constructs a dictionary with the drainage point coordinates as keys and the watershed IDs as values.
    """

    # Open the drainage points file
    ds = ogr.Open(drainage_points_file)
    if ds is None:
        raise ValueError("Could not open drainage points file")

    # Get the layer
    if layer_name is None:
        layer = ds.GetLayer()
    else:
        layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError("Could not open layer in drainage points file")

    # Get the spatial reference of the layer
    spatial_ref = layer.GetSpatialRef()

    # Open the flow direction raster
    fdr_ds = gdal.Open(fdr_filepath)
    if fdr_ds is None:
        raise ValueError("Could not open flow direction raster file")

    # Get the geotransform and inverse geotransform
    geotransform = fdr_ds.GetGeoTransform()
    inv_geotransform = gdal.InvGeoTransform(geotransform)

    # Create a dictionary to store the drainage points
    drainage_points = Dict.empty(UniTuple(int64, 2), int64)

    # Iterate over the features in the layer
    for feature in layer:
        # Get the geometry of the feature
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        # Get the coordinates of the feature
        x, y = geom.GetX(), geom.GetY()

        # Project the coordinates to the raster spatial reference
        if spatial_ref is not None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(spatial_ref.ExportToWkt())
            target_srs = osr.SpatialReference()
            target_srs.ImportFromWkt(fdr_ds.GetProjection())
            transform = osr.CoordinateTransformation(srs, target_srs)
            x, y, _ = transform.TransformPoint(x, y)

        # Convert the coordinates to row and column indices
        col, row = map(int, gdal.ApplyGeoTransform(inv_geotransform, x, y))

        # Add the drainage point to the dictionary
        drainage_points[(row, col)] = 0

    return drainage_points


@njit
def upstream_neighbor_generator(fdr: np.ndarray, row: int, col: int):
    """
    A generator function that yields the upstream neighbor coordinates for a given cell in a flow direction raster.

    Parameters:
    - fdr (np.ndarray): A 2D NumPy array representing the flow direction raster.
                        Each cell contains an integer value indicating the direction of flow.
                        The flow direction values are assumed to be encoded as follows:
                        0: East, 1: Northeast, 2: North, 3: Northwest, 4: West, 5: Southwest, 6: South, 7: Southeast.
    - row (int): The row index of the cell for which to find the upstream neighbors.
    - col (int): The column index of the cell for which to find the upstream neighbors.

    Yields:
    - tuple: A tuple (n_row, n_col) representing the row and column indices of an upstream neighbor cell.

    Description:
    The function iterates over the neighboring cells of the given cell (row, col) using the NEIGHBOR_OFFSETS list.
    For each neighboring cell, it checks if the cell is within the bounds of the flow direction raster.
    If the neighboring cell is within bounds, it retrieves the flow direction value of that cell from the fdr array.
    If the flow direction of the neighboring cell points towards the given cell (i.e., the opposite direction),
    the function yields the row and column indices of the neighboring cell as an upstream neighbor.

    Note:
    - The function assumes that the flow direction values in the fdr array are encoded as defined in
      the constants module.
    """
    rows, cols = fdr.shape
    for i, (d_row, d_col) in enumerate(NEIGHBOR_OFFSETS):
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < rows and 0 <= n_col < cols:
            n_direction = fdr[n_row, n_col]
            if (i + 4) % 8 == n_direction:
                yield n_row, n_col


@njit
def is_outlet(fdr: np.ndarray, row: int, col: int):
    """
    Check if a cell in a flow direction raster is an outlet.

    Parameters:
    - fdr (np.ndarray): A 2D NumPy array representing the flow direction raster.
                        Each cell contains an integer value indicating the direction of flow.
                        The flow direction values are assumed to be encoded as defined in the
                        constants module.
    - row (int): The row index of the cell to check.
    - col (int): The column index of the cell to check.

    Returns:
    - bool: True if the cell is an outlet, False otherwise.

    Description:
    The function checks if a given cell (row, col) in a flow direction raster is an outlet.
    A cell is considered an outlet if it meets any of the following conditions:
    - The next downstream cell, based on the flow direction, is outside the domain of the raster.
    - The next downstream cell has an undefined flow direction (FLOW_DIRECTION_UNDEFINED) or no data
      (FLOW_DIRECTION_NODATA).
    Note that if the current cell itself has an undefined or no data flow direction, it is not considered an outlet.

    The function first checks if the flow direction of the given cell is undefined or no data.
    If so, it immediately returns False, indicating that the cell is not an outlet.

    If the flow direction is valid, the function determines the next downstream cell by using the NEIGHBOR_OFFSETS list.
    It retrieves the row and column offsets corresponding to the flow direction of the given cell.
    The next downstream cell's coordinates are calculated by adding the offsets to the current cell's coordinates.

    The function then checks if the next downstream cell is outside the domain of the raster or if its flow direction
    is undefined or no data. If any of these conditions are met, the function returns True, indicating that the given
    cell is an outlet. Otherwise, it returns False.
    """
    if fdr[row, col] in [FLOW_DIRECTION_UNDEFINED, FLOW_DIRECTION_NODATA]:
        return False
    d_row, d_col = NEIGHBOR_OFFSETS[fdr[row, col]]
    n_row, n_col = row + d_row, col + d_col
    # it is an outlet if the next cell is outside the domain, is nodata, or undefined
    return (
        n_row < 0
        or n_row >= fdr.shape[0]
        or n_col < 0
        or n_col >= fdr.shape[1]
        or fdr[n_row, n_col]
        in [
            FLOW_DIRECTION_UNDEFINED,
            FLOW_DIRECTION_NODATA,
        ]
    )


@njit(parallel=True, nogil=True)
def label_watersheds(
    fdr: np.ndarray,
    drainage_points: dict,
    id_offset: int = 0,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Label watersheds in a flow direction raster based on drainage points.

    Parameters:
    - fdr (np.ndarray): A 2D NumPy array representing the flow direction raster.
                        Each cell contains an integer value indicating the direction of flow.
                        The flow direction values are assumed to be encoded as defined in the
                        constants module.
    - drainage_points (dict): A dictionary containing the drainage points.
                              The keys are tuples (row, col) representing the coordinates of the drainage points,
                              and the values are the corresponding watershed IDs that will be populated by this
                              function.
                              The dictionary will be modified in place to store the watershed IDs.
    - id_offset (int): An offset to be added to the watershed IDs. Default is 0.
    - row_offset (int): An offset to be added to the row coordinates of the drainage points. Default is 0.
    - col_offset (int): An offset to be added to the column coordinates of the drainage points. Default is 0.

    Returns:
    - tuple[np.ndarray, dict]: A tuple containing two elements:
        - watersheds (np.ndarray): A 2D NumPy array with the same shape as fdr, where each cell contains the
                                   watershed ID to which it belongs. Cells that do not belong to any watershed
                                   have a value of 0. Every drainage point or outlet cell and any upstream
                                   cells are assigned a unique watershed ID.
        - local_graph (dict): A dictionary representing the local graph of the watersheds.
                              The keys are the watershed IDs, and the values are the watershed IDs of the
                              downstream neighbors.

    Description:
    The function labels watersheds in a flow direction raster based on given drainage points.
    It performs a parallel breadth-first search (BFS) starting from each outlet cell in the raster.
    The BFS traverses upstream cells using the flow direction information to assign watershed IDs.

    The function initializes a watersheds array with the same shape as fdr and fills it with zeros.

    The function iterates over each cell in the fdr array using parallel processing.
    For each cell, it checks if the cell is an outlet using the is_outlet function.
    If the cell is an outlet, it assigns a unique watershed ID to the cell and starts a BFS from that cell.

    During the BFS, the function uses a queue to store the cells to be processed.
    It pops a cell from the queue and checks if the cell's coordinates (adjusted by the offsets) match any
    drainage point.
    If a match is found, it updates the watershed ID of the cell to the corresponding value from the
    drainage_points dictionary.

    If the cell is not an outlet and its watershed ID is different from the previous watershed ID,
    the function adds an edge to the local graph, connecting the current watershed ID to the previous watershed ID.

    The function then iterates over the upstream neighbors of the current cell using the
    upstream_neighbor_generator function.
    It assigns the current watershed ID to each upstream neighbor and pushes them into the queue for further processing.

    Finally, the function returns the watersheds array and the local_graph dictionary.
    """
    watersheds = np.zeros_like(fdr, dtype=np.int64)
    rows, cols = fdr.shape
    # instead of a dictionary, we use a 2D array to store the downstream watersheds
    # since dictionary appends are not thread-safe
    downstream_watersheds = np.zeros_like(fdr, dtype=np.int64)
    num_cells = rows * cols
    for index in prange(num_cells):  # pylint: disable=not-an-iterable
        row = index // cols
        col = index % cols
        if is_outlet(fdr, row, col):
            watershed_id = row * cols + col + 1 + id_offset
            queue = Queue([GridCell(row, col, watershed_id)])
            watersheds[row, col] = watershed_id

            while queue:
                cell = queue.pop()
                r = cell.row
                c = cell.col
                watershed_id = cell.value
                if (r + row_offset, c + col_offset) in drainage_points:
                    prev_watershed_id = watershed_id
                    watershed_id = r * cols + c + 1 + id_offset
                    # if this is not an outlet, add an edge to the local graph
                    if prev_watershed_id != watershed_id:
                        downstream_watersheds[r, c] = prev_watershed_id
                    drainage_points[(r + row_offset, c + col_offset)] = watershed_id
                    watersheds[r, c] = watershed_id
                for n_row, n_col in upstream_neighbor_generator(fdr, r, c):
                    watersheds[n_row, n_col] = watershed_id
                    queue.push(GridCell(n_row, n_col, watershed_id))

    # Create local graph from downstream_watersheds and drainage_points
    local_graph = Dict.empty(int64, int64)
    for (r, c), watershed_id in drainage_points.items():
        if (
            r - row_offset < 0
            or r - row_offset >= rows
            or c - col_offset < 0
            or c - col_offset >= cols
        ):
            continue
        downstream_id = downstream_watersheds[r - row_offset, c - col_offset]
        if downstream_id != 0:
            local_graph[watershed_id] = downstream_id

    return watersheds, local_graph


def label_watersheds_from_file(
    fdr_filepath: str,
    drainage_points_file: str,
    output_file: str,
    all_basins=True,
    layer_name=None,
):
    """
    Label watersheds in a flow direction raster based on drainage points from an OGR compatible file.

    Parameters:
    - fdr_filepath (str): The path to the flow direction raster file.
    - drainage_points_file (str): The path to the drainage points file.
                                  The file should be in an OGR compatible format (e.g., Shapefile, GeoPackage).
    - output_file (str): The path to save the labeled watersheds raster.
    - all_basins (bool): If True, label all basins. If False, only label basins connected to drainage points.
        Default is True.
    - layer_name (str): The name of the layer in the drainage points file to read.
                        If None, the first layer in the file will be used. Default is None

    Returns:
    - None

    Description:
    The function reads the flow direction raster from the fdr_filepath using the GDAL library.
    It reads the drainage points from the drainage_points_file using the drainage_points_from_file function.
    The function then calls the label_watersheds function to label the watersheds based on the flow direction raster
    and the drainage points.
    """
    fdr_ds = gdal.Open(fdr_filepath)
    if fdr_ds is None:
        raise ValueError("Could not open flow direction raster file")

    fdr = fdr_ds.GetRasterBand(1).ReadAsArray()
    drainage_points = drainage_points_from_file(
        fdr_filepath, drainage_points_file, layer_name
    )

    watersheds, graph = label_watersheds(fdr, drainage_points)
    if not all_basins:
        # remove any label not in drainage_points values
        unique_labels = np.unique(watersheds)
        for label in unique_labels:
            if label not in drainage_points.values():
                watersheds[watersheds == label] = 0
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

    basin_polygons_filepath = output_file.replace(".tif", ".gpkg")
    chunk_size = max(fdr.shape[0], fdr.shape[1])
    create_basin_polygons(
        out_band,
        graph,
        chunk_size,
        basin_polygons_filepath,
        out_ds.GetGeoTransform(),
        out_ds.GetProjection(),
    )

    out_band = None
    out_ds = None
