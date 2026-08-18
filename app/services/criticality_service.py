import networkx as nx

from pipeline.criticality import (
    get_largest_component,
    build_critical_node_table,
    build_critical_edge_table,
    identify_bridges_and_articulations,
    TOP_N,
)
from pipeline.corridors import (
    build_critical_subgraph,
    aggregate_corridors,
    TOP_EDGES,
)



def get_criticality(graph):

    # Work on the largest connected component
    H = get_largest_component(graph)

    # Critical nodes
    node_table = build_critical_node_table(H)

    # Critical roads
    edge_table = build_critical_edge_table(H)

    # Bridges and articulation points
    bridge_df, articulation_df = (
        identify_bridges_and_articulations(H)
    )

    return {
        "critical_roads": (
            edge_table.head(TOP_N)
        ),
        "critical_nodes": (
            node_table.head(TOP_N)
        ),
        "bridges": bridge_df,
        "articulation_points": articulation_df,
    }

def get_criticality_response(graph):

    result = get_criticality(graph)

    critical_roads = (
        result["critical_roads"]
        .to_dict(orient="records")
    )

    critical_nodes = (
        result["critical_nodes"]
        .to_dict(orient="records")
    )

    return {
        "critical_roads": critical_roads,
        "critical_nodes": critical_nodes,
        "bridge_count": len(
            result["bridges"]
        ),
        "articulation_point_count": len(
            result["articulation_points"]
        ),
    }


def get_corridors(
    graph,
    critical_roads,
    as_records=True,
):
    critical_roads = (
        critical_roads
        .sort_values(
            "edge_betweenness",
            ascending=False,
        )
        .head(TOP_EDGES)
        .copy()
    )

    H, critical_roads = build_critical_subgraph(
        graph,
        critical_roads,
    )

    (
        corridor_df,
        corridor_geometries,
    ) = aggregate_corridors(
        graph,
        H,
        critical_roads,
    )

    if corridor_df.empty:
        return (
            []
            if as_records
            else corridor_df
        )

    corridor_df = (
        corridor_df
        .sort_values(
            "criticality_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    corridor_df["corridor_id"] = range(
        1,
        len(corridor_df) + 1,
    )

    # IMPORTANT:
    # preserve corridor geometry
    corridor_df["geometry"] = (
        corridor_geometries
    )

    if as_records:
        return corridor_df.to_dict(
            orient="records"
        )

    return corridor_df
