import math

import networkx as nx


def nearest_node(
    graph,
    x,
    y,
):
    best_node = None
    best_distance = float("inf")

    for node, data in graph.nodes(data=True):

        node_x = data.get("x")
        node_y = data.get("y")

        if node_x is None or node_y is None:
            continue

        distance = math.hypot(
            node_x - x,
            node_y - y,
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    if best_node is None:
        raise ValueError(
            "Could not find a nearby graph node."
        )

    return best_node


def find_route(
    graph,
    start_x,
    start_y,
    end_x,
    end_y,
):
    start_node = nearest_node(
        graph,
        start_x,
        start_y,
    )

    end_node = nearest_node(
        graph,
        end_x,
        end_y,
    )

    if start_node == end_node:
        return {
            "start_node": start_node,
            "end_node": end_node,
            "path": [start_node],
            "distance_m": 0.0,
        }

    try:

        path = nx.shortest_path(
            graph,
            source=start_node,
            target=end_node,
            weight="length_m",
        )

    except nx.NetworkXNoPath:

        raise ValueError(
            "No route exists between the selected points."
        )

    distance = nx.path_weight(
        graph,
        path,
        weight="length_m",
    )

    return {
        "start_node": start_node,
        "end_node": end_node,
        "path": path,
        "distance_m": float(distance),
    }