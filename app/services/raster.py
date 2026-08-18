import rasterio 

def inspect_band(path):
    """
    Inspect the given path to check if it is a valid raster file.
    Args:
        path (str): The file path to inspect."""
    try:
        with rasterio.open(path) as src:
            if src.count != 4:
                raise ValueError("The raster file must have exactly 4 bands (B02, B03, B04, B08).")
            
            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
                "resolution": src.res,
                "dtype": str(src.dtypes[0]),
                
            }
    except rasterio.errors.RasterioIOError as e:
        raise ValueError("the uploaded file is not a valid geotiff") from e
