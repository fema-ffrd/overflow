from osgeo import gdal

from overflow.fix_flats.core.fix_flats import d8_masked_flow_dirs
from overflow.util.raster import RasterChunk, raster_chunker


def update_fdr(
    dem_band: gdal.Band,
    fdr_band: gdal.Band,
    fixed_fdr_band: gdal.Band,
    flat_mask_band: gdal.Band,
    chunk_size: int,
    progress_callback=None,
):
    # update fdr using d8_masked_flow_dirs
    for fdr_tile in raster_chunker(
        fdr_band, chunk_size, 1, progress_callback=progress_callback
    ):
        dem_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        dem_tile.read(dem_band)
        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        flat_mask_tile.read(flat_mask_band)
        d8_masked_flow_dirs(dem_tile.data, flat_mask_tile.data, fdr_tile.data)
        fdr_tile.write(fixed_fdr_band)
    fixed_fdr_band.FlushCache()
