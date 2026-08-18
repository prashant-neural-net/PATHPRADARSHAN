# # # # # # # # # ### this file is used to test different functions

# # # # # # # # # ## combine_bands function
# # # # # # # # # from app.services.combine_bands import combine_bands
# # # # # # # # from pathlib import Path

# # # # # # # # # path1 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B02_(Raw).tiff")
# # # # # # # # # path2 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B03_(Raw).tiff")
# # # # # # # # # path3 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B04_(Raw).tiff")
# # # # # # # # # path4 = Path("tests/test_data/2026-05-12-00_00_2026-05-12-23_59_Sentinel-2_L2A_B08_(Raw).tiff")


# # # # # # # # # # output_path = combine_bands(path1, path2, path3, path4)



# # # # # # # # # ###### testing the dimension of the output file
# # # # # # # # # from app.services.raster import inspect_band

# # # # # # # # # path = Path("/home/prashanttripathi19042004/PATHPRADARSHAN/tests/output.tiff")

# # # # # # # # # res = inspect_band(path)

# # # # # # # # # print(res)

# # # # # # # # from app.services.inference import read_raster

# # # # # # # # image, profile, height, width = read_raster("/home/prashanttripathi19042004/PATHPRADARSHAN/tests/output.tiff")

# # # # # # # # print("image shape", image.shape)
# # # # # # # # print("height", height)
# # # # # # # # print("width", width)
# # # # # # # # print("CRS", profile["crs"])

# # # # # # # ### inference test

# # # import numpy as np

# # # from app.services.inference import (
# # #     load_model,
# # #     read_raster,
# # #     predict_patches
# # # )


# # # model = load_model()
# # # image, profile, height, width = read_raster(
# # #     "/home/prashanttripathi19042004/PATHPRADARSHAN/tests/test_data/image_0007.tif"
# # # )

# # # print("image min",image.min())
# # # print("input max", image.max())
# # # print("input mean:", image.mean())

# # # probability_map = predict_patches(
# # #     image,
# # #     model
# # # )

# # # print("img shape", image.shape)
# # # print("prob shape", probability_map.shape)
# # # print("Min: ",probability_map.min())
# # # print("Max: ",probability_map.max())
# # # print("mean: ", probability_map.mean())
# # # print("median: ", np.median(probability_map))


# # # # # ### saving inference result

# # from pathlib import Path

# # from app.services.inference import run_inference

# # input_path = Path("/home/prashanttripathi19042004/PATHPRADARSHAN/data/uploads/image_0008.tif")

# # probability_path = Path(
# #     "data/results/combined_probability.tif"
# # )

# # mask_path = Path(
# #     "data/results/combined_road_mask.tif"
# # )

# # result = run_inference(
# #     input_path,
# #     probability_path,
# #     mask_path
# # )

# # print("Inference completed!")
# # print("probability:", result["probability_path"])
# # print("Road  mask:", result["mask_path"])

# # import osmnx as ox

# # bbox = (
# #     77.48942773344164,   # left
# #     13.030979357514939,  # bottom
# #     77.51324891970732,   # right
# #     13.054334447240869,  # top
# # )

# # print("Downloading OSM roads...")

# # roads = ox.features.features_from_bbox(
# #     bbox,
# #     tags={"highway": True},
# # )

# # print("Downloaded features:", len(roads))

# # # Keep only line geometries
# # roads = roads[
# #     roads.geometry.geom_type.isin(
# #         [
# #             "LineString",
# #             "MultiLineString",
# #         ]
# #     )
# # ].copy()

# # print("Road geometries:", len(roads))

# # # Keep useful columns if they exist
# # columns = [
# #     "osmid",
# #     "highway",
# #     "name",
# #     "geometry",
# # ]

# # columns = [
# #     col
# #     for col in columns
# #     if col in roads.columns
# # ]

# # roads = roads[columns]

# # # Rename to match your fusion.py
# # if "osmid" in roads.columns:
# #     roads = roads.rename(
# #         columns={
# #             "osmid": "osm_id"
# #         }
# #     )

# # roads.to_file(
# #     "data/osm_roads.gpkg",
# #     layer="roads",
# #     driver="GPKG",
# # )

# # print(
# #     "Saved: data/osm_roads.gpkg"
# # )
# from pathlib import Path

# from pipeline.fusion import (
#     build_road_confidence_gdf,
# )


# osm_path = Path(
#     "data/osm_roads.gpkg"
# )

# raster_path = Path(
#     "results/test_probability.tif"
# )

# output_path = Path(
#     "data/road_edge_confidence.gpkg"
# )


# build_road_confidence_gdf(
#     osm_path=str(osm_path),
#     raster_path=str(raster_path),
#     output_path=str(output_path),
# )

# print(
#     "Fusion completed!"
# )

import geopandas as gpd

from pipeline.graph import build_osm_graph


CONFIDENCE_PATH = (
    "data/road_edge_confidence.gpkg"
)


print("Loading confidence roads...")

gdf = gpd.read_file(
    CONFIDENCE_PATH,
    layer="road_confidence",
)

print(
    "Roads loaded:",
    len(gdf)
)

print(
    "CRS:",
    gdf.crs
)

print("Building graph...")

graph = build_osm_graph(
    gdf
)

print()
print("==============================")
print("GRAPH TEST")
print("==============================")

print(
    "Nodes:",
    graph.number_of_nodes()
)

print(
    "Edges:",
    graph.number_of_edges()
)

total_length = sum(
    data.get("length_m", 0.0)
    for _, _, data
    in graph.edges(data=True)
)

print(
    "Total road length:",
    total_length,
    "m"
)