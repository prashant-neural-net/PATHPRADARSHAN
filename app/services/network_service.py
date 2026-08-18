import geopandas as gpd

from pipeline.graph import build_osm_graph


CONFIDENCE_PATH = (
    "data/road_edge_confidence.gpkg"
)


def get_network_graph():

    gdf = gpd.read_file(
        CONFIDENCE_PATH,
        layer="road_confidence",
    )

    graph = build_osm_graph(
        gdf
    )

    return graph


def get_network_summary(graph):

    # graph = get_network_graph()

    total_junctions = (
        graph.number_of_nodes()
    )

    total_roads = (
        graph.number_of_edges()
    )

    total_road_length_m = sum(
        data.get(
            "length_m",
            0.0
        )
        for _, _, data
        in graph.edges(data=True)
    )

    return {
        "total_junctions":
            total_junctions,

        "total_roads":
            total_roads,

        "total_road_length_m":
            total_road_length_m,

        "total_road_length_km":
            total_road_length_m / 1000,
    }