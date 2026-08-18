from pathlib import Path
import rasterio

output_path = Path("/home/prashanttripathi19042004/PATHPRADARSHAN/tests/output.tiff")
if not output_path.exists():
    output_path.touch()

def combine_bands(
        b02_path: str | Path,
        b03_path: str | Path,
        b04_path: str | Path,
        b08_path: str | Path
):
    input_paths = [
        Path(b02_path),
        Path(b03_path),
        Path(b04_path),
        Path(b08_path)
    ]

    with rasterio.open(input_paths[0]) as src:
        profile = src.profile.copy()

        profile.update(
            count=4,
            dtype="float32"
        ) 
        with rasterio.open(output_path, "w", **profile) as output:
            for band_number, filename in enumerate(input_paths, start=1):
                with rasterio.open(filename) as band:
                    if (
                        band.width != src.width
                        or band.height != src.height
                        or band.transform != src.transform
                        or band.crs != src.crs
                    ):
                        raise ValueError(f"{filename} does not match B02 geometry")
                    
                    data = band.read(1).astype("float32")

                    output.write(data, band_number)

    print(output_path)

