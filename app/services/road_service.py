from ..services.network_service import get_network_graph


def get_road(road_id):

    graph = get_network_graph()

    for u, v, data in graph.edges(data=True):

        if data.get("edge_id") == road_id:

            return {
                "road_id": road_id,
                "source": u,
                "target": v,
                "osm_id": data.get("osm_id"),
                "highway": data.get("highway"),
                "length_m": data.get("length_m"),
                "mean_prob": data.get("mean_prob"),
                "median_prob": data.get("median_prob"),
                "min_prob": data.get("min_prob"),
                "confidence": data.get("confidence"),
                "status": data.get("status"),
            }

    return None