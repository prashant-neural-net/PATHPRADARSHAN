# # ### this file is used to test different functions

# # ## combine_bands function
# # from app.services.combine_bands import combine_bands
# from pathlib import Path

# # path1 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B02_(Raw).tiff")
# # path2 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B03_(Raw).tiff")
# # path3 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B04_(Raw).tiff")
# # path4 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B08_(Raw).tiff")


# # output_path = combine_bands(path1, path2, path3, path4)



# ###### testing the dimension of the output file
# from app.services.raster import inspect_band

# path = Path("/home/prashanttripathi19042004/PATHPRADARSHAN/tests/output.tiff")

# res = inspect_band(path)

# print(res)

from app.services.inference import read_raster

image, profile, height, width = read_raster("/home/prashanttripathi19042004/PATHPRADARSHAN/tests/output.tiff")

print("image shape", image.shape)
print("height", height)
print("width", width)
print("CRS", profile["crs"])