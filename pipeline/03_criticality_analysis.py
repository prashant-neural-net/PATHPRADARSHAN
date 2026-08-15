#!/usr/bin/env python3

"""
Large-scale criticality analysis for the OSM + V5 road graph.

The graph topology comes from OSM.
Satellite-derived V5 confidence is retained as edge evidence.

For scalability:
- Exact degree centrality
- Approximate node betweenness
- Approximate weighted node betweenness
- Approximate edge betweenness
- Exact bridges
- Exact articulation points

Centrality is calculated on the largest connected component because
99%+ of the network belongs to that component.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


DEFAULT_GRAPH = "bengaluru_road_graph.gpickle"
DEFAULT_CONFIDENCE = "road_edge_confidence.gpkg"
DEFAULT_CRITICAL_ROADS = "critical_roads.csv"
DEFAULT_CRITICAL_NODES = "critical_nodes.csv"
DEFAULT_CRITICAL_GPKG = "top_critical_roads.gpkg"

# ============================================================
# CONFIG
# ============================================================

BETWEENNESS_K = 100
RANDOM_SEED = 42
TOP_N = 1000


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    with open(path, "rb") as handle:
        return pickle.load(handle)


# ============================================================
# GRAPH SUMMARY
# ============================================================

def graph_summary(G):

    components = [
        len(c)
        for c in nx.connected_components(G)
    ]

    largest = max(
        components,
        default=0
    )

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(components),
        "largest_component": largest,
    }


# ============================================================
# LARGEST COMPONENT
# ============================================================

def get_largest_component(G):

    largest_nodes = max(
        nx.connected_components(G),
        key=len
    )

    H = G.subgraph(
        largest_nodes
    ).copy()

    return H


# ============================================================
# EDGE BETWEenness
# ============================================================

def build_critical_edge_table(G):

    print()
    print(
        "Calculating approximate edge betweenness..."
    )

    edge_betweenness = (
        nx.edge_betweenness_centrality(
            G,
            k=BETWEENNESS_K,
            normalized=True,
            weight="length_m",
            seed=RANDOM_SEED,
        )
    )

    rows = []

    for (
        u,
        v
    ), score in sorted(
        edge_betweenness.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        edge = G[u][v]

        rows.append({

            "source":
                u,

            "target":
                v,

            "edge_betweenness":
                float(score),

            "length_m":
                float(
                    edge.get(
                        "length_m",
                        0.0
                    )
                ),

            "confidence":
                float(
                    edge.get(
                        "confidence",
                        0.0
                    )
                ),

            "status":
                edge.get(
                    "status",
                    "UNCERTAIN"
                ),

            "highway":
                edge.get(
                    "highway",
                    "unknown"
                ),

            "osm_id":
                edge.get(
                    "osm_id",
                    -1
                ),

            "edge_id":
                edge.get(
                    "edge_id"
            ),

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# NODE CENTRALITY
# ============================================================

def build_critical_node_table(G):

    print()
    print("Calculating degree centrality...")

    degree = nx.degree_centrality(G)

    print(
        f"Calculating approximate node betweenness "
        f"(k={BETWEENNESS_K})..."
    )

    betweenness = nx.betweenness_centrality(
        G,
        k=BETWEENNESS_K,
        normalized=True,
        seed=RANDOM_SEED,
    )

    rows = []

    for node in G.nodes:

        rows.append({
            "node_id": node,

            "x": float(
                G.nodes[node].get("x", 0.0)
            ),

            "y": float(
                G.nodes[node].get("y", 0.0)
            ),

            "degree":
                int(G.degree(node)),

            "degree_centrality":
                float(degree.get(node, 0.0)),

            "betweenness_centrality":
                float(
                    betweenness.get(node, 0.0)
                ),

            "node_type":
                G.nodes[node].get(
                    "node_type",
                    "unknown"
                ),
        })

    df = pd.DataFrame(rows)

    return df.sort_values(
        "betweenness_centrality",
        ascending=False
    )

# ============================================================
# BRIDGES + ARTICULATION POINTS
# ============================================================

def identify_bridges_and_articulations(G):

    print()
    print(
        "Finding bridge edges..."
    )

    bridges = []

    for u, v in nx.bridges(G):

        data = G[u][v]

        bridges.append({

            "source":
                u,

            "target":
                v,

            "length_m":
                float(
                    data.get(
                        "length_m",
                        0.0
                    )
                ),

            "status":
                data.get(
                    "status",
                    "UNCERTAIN"
                ),

            "confidence":
                float(
                    data.get(
                        "confidence",
                        0.0
                    )
                ),

            "osm_id":
                data.get(
                    "osm_id",
                    -1
                ),

            "edge_id":
                data.get(
                    "edge_id"
                ),
        })

    print(
        "Finding articulation points..."
    )

    articulation = []

    for node in nx.articulation_points(
        G
    ):

        articulation.append({

            "node_id":
                node,

            "x":
                float(
                    G.nodes[node].get(
                        "x",
                        0.0
                    )
                ),

            "y":
                float(
                    G.nodes[node].get(
                        "y",
                        0.0
                    )
                ),
        })

    return (
        pd.DataFrame(bridges),
        pd.DataFrame(articulation)
    )


# ============================================================
# EDGE → GEODATAFRAME
# ============================================================

def edge_rows_to_geodataframe(
    rows,
    G
):

    features = []

    for _, row in rows.iterrows():

        u = int(
            row["source"]
        )

        v = int(
            row["target"]
        )

        if not G.has_edge(
            u,
            v
        ):
            continue

        geom = G[u][v].get(
            "geometry"
        )

        if geom is None:
            continue

        features.append({

            **row.to_dict(),

            "geometry":
                geom,
        })

    if not features:

        return gpd.GeoDataFrame(
            [],
            geometry="geometry",
            crs="EPSG:32643"
        )

    return gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs="EPSG:32643"
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    graph_path,
    roads_conf_path,
    critical_roads_csv,
    critical_nodes_csv,
    critical_gpkg_path,
):

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    print(
        "Loading graph..."
    )

    G = load_graph(
        graph_path
    )

    summary = graph_summary(
        G
    )

    print()
    print("==============================")
    print("CRITICALITY ANALYSIS")
    print("==============================")

    print(
        "Graph summary:",
        summary
    )

    # --------------------------------------------------------
    # Largest connected component
    # --------------------------------------------------------

    print()
    print(
        "Extracting largest connected component..."
    )

    H = get_largest_component(
        G
    )

    print(
        "Largest component nodes:",
        H.number_of_nodes()
    )

    print(
        "Largest component edges:",
        H.number_of_edges()
    )

    # --------------------------------------------------------
    # Node analysis
    # --------------------------------------------------------

    node_table = (
        build_critical_node_table(
            H
        )
    )

    # --------------------------------------------------------
    # Edge analysis
    # --------------------------------------------------------

    edge_table = (
        build_critical_edge_table(
            H
        )
    )

    # --------------------------------------------------------
    # Bridges / articulation points
    # --------------------------------------------------------

    bridge_df, articulation_df = (
        identify_bridges_and_articulations(
            H
        )
    )

    # --------------------------------------------------------
    # Top results
    # --------------------------------------------------------

    top_edges = edge_table.head(
        TOP_N
    ).copy()

    top_nodes = node_table.head(
        TOP_N
    ).copy()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    top_edges.to_csv(
        critical_roads_csv,
        index=False
    )

    top_nodes.to_csv(
        critical_nodes_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save critical roads GeoPackage
    # --------------------------------------------------------

    critical_geo = (
        edge_rows_to_geodataframe(
            top_edges,
            H
        )
    )

    if not critical_geo.empty:

        critical_geo.to_file(
            critical_gpkg_path,
            driver="GPKG"
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("==============================")
    print("TOP 10 CRITICAL ROAD EDGES")
    print("==============================")

    for rank, (_, row) in enumerate(
        top_edges.iterrows(),
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{int(row['source'])} -> "
            f"{int(row['target'])} | "
            f"Betweenness: "
            f"{row['edge_betweenness']:.6f} | "
            f"Length: "
            f"{row['length_m']:.1f} m | "
            f"Confidence: "
            f"{row['confidence']:.3f} | "
            f"Status: "
            f"{row['status']}"
        )
    print()
    print("==============================")
    print("TOP 10 CRITICAL NODES")
    print("==============================")

    for rank, (_, row) in enumerate(
        top_nodes.iterrows(),
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Node {int(row['node_id'])} | "
            f"Betweenness: "
            f"{row['betweenness_centrality']:.6f} | "
            f"Degree: "
            f"{int(row['degree'])}"
        )

    print()
    print("==============================")
    print("STRUCTURAL VULNERABILITY")
    print("==============================")

    print(
        "Bridge edges:",
        len(bridge_df)
    )

    print(
        "Articulation points:",
        len(articulation_df)
    )

    print()
    print("==============================")
    print("CRITICALITY ANALYSIS COMPLETE")
    print("==============================")

    print(
        "Saved:",
        critical_roads_csv
    )

    print(
        "Saved:",
        critical_nodes_csv
    )

    print(
        "Saved:",
        critical_gpkg_path
    )

    return (
        top_edges,
        top_nodes,
        bridge_df,
        articulation_df
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compute scalable centrality "
            "and criticality metrics."
        )
    )

    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH
    )

    parser.add_argument(
        "--confidence",
        default=DEFAULT_CONFIDENCE
    )

    parser.add_argument(
        "--roads-output",
        default=DEFAULT_CRITICAL_ROADS
    )

    parser.add_argument(
        "--nodes-output",
        default=DEFAULT_CRITICAL_NODES
    )

    parser.add_argument(
        "--critical-gpkg",
        default=DEFAULT_CRITICAL_GPKG
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    run_analysis(

        graph_path=args.graph,

        roads_conf_path=args.confidence,

        critical_roads_csv=args.roads_output,

        critical_nodes_csv=args.nodes_output,

        critical_gpkg_path=args.critical_gpkg,
    )