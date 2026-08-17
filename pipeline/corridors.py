#!/usr/bin/env python3

"""
Aggregate critical OSM road segments into meaningful road corridors.

Inputs:
    bengaluru_road_graph.gpickle
    critical_roads.csv

Outputs:
    critical_corridors.csv
    critical_corridors.gpkg
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.ops import unary_union


DEFAULT_GRAPH = "bengaluru_road_graph.gpickle"
DEFAULT_CRITICAL_ROADS = "critical_roads.csv"
DEFAULT_OUTPUT_CSV = "critical_corridors.csv"
DEFAULT_OUTPUT_GPKG = "critical_corridors.gpkg"

# Number of top critical edge records to aggregate
TOP_EDGES = 1000


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD CRITICAL EDGES
# ============================================================

def load_critical_edges(path):

    df = pd.read_csv(path)

    required = {
        "source",
        "target",
        "edge_betweenness",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing columns in critical_roads.csv: {missing}"
        )

    return df


# ============================================================
# BUILD CRITICAL EDGE SUBGRAPH
# ============================================================

def build_critical_subgraph(
    G,
    critical_df
):

    critical_df = critical_df.copy()

    critical_df = critical_df.sort_values(
        "edge_betweenness",
        ascending=False
    ).head(
        TOP_EDGES
    )

    critical_edges = []

    for _, row in critical_df.iterrows():

        u = int(row["source"])
        v = int(row["target"])

        if not G.has_edge(u, v):
            continue

        critical_edges.append(
            (u, v)
        )

    H = nx.Graph()

    H.add_edges_from(
        critical_edges
    )

    return H, critical_df


# ============================================================
# CORRIDOR AGGREGATION
# ============================================================

def aggregate_corridors(
    G,
    H,
    critical_df
):

    # Map edge pair → criticality row
    critical_lookup = {}

    for _, row in critical_df.iterrows():

        key = (
            int(row["source"]),
            int(row["target"])
        )

        reverse_key = (
            key[1],
            key[0]
        )

        critical_lookup[key] = row
        critical_lookup[reverse_key] = row

    components = list(
        nx.connected_components(H)
    )

    print(
        "Critical edge groups:",
        len(components)
    )

    records = []

    corridor_geometries = []

    corridor_id = 1

    for component in components:

        component_edges = []

        for u, v in H.subgraph(
            component
        ).edges():

            if G.has_edge(u, v):

                component_edges.append(
                    (u, v)
                )

        if not component_edges:
            continue

        # ----------------------------------------------------
        # Collect attributes
        # ----------------------------------------------------

        lengths = []
        betweenness = []
        confidences = []
        min_confidences = []

        highway_types = []
        statuses = []

        geometries = []
        osm_ids = []

        for u, v in component_edges:

            data = G[u][v]

            lengths.append(
                float(
                    data.get(
                        "length_m",
                        0.0
                    )
                )
            )

            row = critical_lookup.get(
                (u, v)
            )

            if row is not None:

                betweenness.append(
                    float(
                        row[
                            "edge_betweenness"
                        ]
                    )
                )

            confidences.append(
                float(
                    data.get(
                        "confidence",
                        0.0
                    )
                )
            )

            min_confidences.append(
                float(
                    data.get(
                        "confidence",
                        0.0
                    )
                )
            )

            highway = data.get(
                "highway",
                "unknown"
            )

            if highway is not None:

                highway_types.append(
                    str(highway)
                )

            status = data.get(
                "status",
                "UNCERTAIN"
            )

            if status is not None:

                statuses.append(
                    str(status)
                )

            geometry = data.get(
                "geometry"
            )

            if geometry is not None:

                geometries.append(
                    geometry
                )

            osm_id = data.get(
                "osm_id"
            )

            if osm_id is not None:

                osm_ids.append(
                    osm_id
                )

        if not lengths:
            continue

        # ----------------------------------------------------
        # Corridor statistics
        # ----------------------------------------------------

        total_length = sum(
            lengths
        )

        mean_betweenness = (
            sum(betweenness)
            /
            len(betweenness)
            if betweenness
            else 0.0
        )

        max_betweenness = (
            max(betweenness)
            if betweenness
            else 0.0
        )

        mean_confidence = (
            sum(confidences)
            /
            len(confidences)
            if confidences
            else 0.0
        )

        min_confidence = (
            min(min_confidences)
            if min_confidences
            else 0.0
        )

        # ----------------------------------------------------
        # Dominant status
        # ----------------------------------------------------

        if statuses:

            status_counts = (
                pd.Series(
                    statuses
                ).value_counts()
            )

            dominant_status = (
                status_counts.index[0]
            )

        else:

            dominant_status = (
                "UNCERTAIN"
            )

        # ----------------------------------------------------
        # Dominant highway type
        # ----------------------------------------------------

        if highway_types:

            highway_counts = (
                pd.Series(
                    highway_types
                ).value_counts()
            )

            dominant_highway = (
                highway_counts.index[0]
            )

        else:

            dominant_highway = (
                "unknown"
            )

        # ----------------------------------------------------
        # Criticality score
        # ----------------------------------------------------
        #
        # High betweenness + long corridor +
        # lower satellite confidence = higher concern.
        #
        # This is a prioritization score, not
        # ground-truth probability.
        # ----------------------------------------------------

        criticality_score = (
            mean_betweenness
            *
            (
                1.0
                +
                min(
                    total_length / 1000.0,
                    5.0
                )
            )
            *
            (
                1.0
                +
                (
                    1.0
                    -
                    mean_confidence
                )
            )
        )

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        corridor_geometry = None

        if geometries:

            try:

                corridor_geometry = (
                    unary_union(
                        geometries
                    )
                )

            except Exception:

                corridor_geometry = (
                    geometries[0]
                )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({

            "corridor_id":
                corridor_id,

            "segment_count":
                len(component_edges),

            "total_length_m":
                total_length,

            "mean_betweenness":
                mean_betweenness,

            "max_betweenness":
                max_betweenness,

            "mean_confidence":
                mean_confidence,

            "min_confidence":
                min_confidence,

            "dominant_status":
                dominant_status,

            "dominant_highway":
                dominant_highway,

            "criticality_score":
                criticality_score,

            "osm_segment_count":
                len(
                    set(
                        osm_ids
                    )
                ),
        })

        corridor_geometries.append(
            corridor_geometry
        )

        corridor_id += 1

    corridor_df = pd.DataFrame(
        records
    )

    return (
        corridor_df,
        corridor_geometries
    )


# ============================================================
# MAIN
# ============================================================

def run(
    graph_path,
    critical_roads_path,
    output_csv,
    output_gpkg
):

    print(
        "Loading graph..."
    )

    G = load_graph(
        graph_path
    )

    print(
        "Graph nodes:",
        G.number_of_nodes()
    )

    print(
        "Graph edges:",
        G.number_of_edges()
    )

    print(
        "Loading critical roads..."
    )

    critical_df = load_critical_edges(
        critical_roads_path
    )

    print(
        "Critical edge records:",
        len(critical_df)
    )

    # --------------------------------------------------------
    # Build critical subgraph
    # --------------------------------------------------------

    print()
    print(
        "Building critical-edge subgraph..."
    )

    H, critical_df = (
        build_critical_subgraph(
            G,
            critical_df
        )
    )

    print(
        "Critical edges used:",
        H.number_of_edges()
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print()
    print(
        "Aggregating critical corridors..."
    )

    (
        corridor_df,
        corridor_geometries
    ) = aggregate_corridors(
        G,
        H,
        critical_df
    )

    if corridor_df.empty:

        raise RuntimeError(
            "No critical corridors were generated."
        )

    # --------------------------------------------------------
    # Sort by criticality
    # --------------------------------------------------------

    corridor_df = (
        corridor_df
        .sort_values(
            "criticality_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # Re-number after sorting
    corridor_df[
        "corridor_id"
    ] = range(
        1,
        len(corridor_df) + 1
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    corridor_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # GeoPackage
    # --------------------------------------------------------

    geometry_map = {}

    # We need to recreate the geometries in
    # the same sorted order.
    #
    # Recalculate from original corridor order.
    _, original_geometries = (
        aggregate_corridors(
            G,
            H,
            critical_df
        )
    )

    sorted_indices = (
        corridor_df.index
    )

    geometries_sorted = [
        original_geometries[i]
        for i in sorted_indices
        if i < len(original_geometries)
    ]

    geo_df = corridor_df.copy()

    geo_df[
        "geometry"
    ] = geometries_sorted

    geo_df = gpd.GeoDataFrame(
        geo_df,
        geometry="geometry",
        crs="EPSG:32643"
    )

    geo_df.to_file(
        output_gpkg,
        driver="GPKG"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("==============================")
    print("CRITICAL CORRIDOR AGGREGATION")
    print("==============================")

    print(
        "Critical segments used:",
        H.number_of_edges()
    )

    print(
        "Corridors:",
        len(corridor_df)
    )

    print()
    print(
        "TOP 10 CRITICAL CORRIDORS"
    )

    print(
        "------------------------------------------------------------"
    )

    for rank, (_, row) in enumerate(
        corridor_df.head(10).iterrows(),
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Corridor {int(row['corridor_id'])} | "
            f"Segments: {int(row['segment_count'])} | "
            f"Length: {row['total_length_m']:.1f} m | "
            f"Betweenness: {row['mean_betweenness']:.6f} | "
            f"Confidence: {row['mean_confidence']:.3f} | "
            f"Status: {row['dominant_status']} | "
            f"Score: {row['criticality_score']:.4f}"
        )

    print()
    print(
        "Saved:",
        output_csv
    )

    print(
        "Saved:",
        output_gpkg
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate critical OSM road segments "
            "into road corridors."
        )
    )

    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH
    )

    parser.add_argument(
        "--critical-roads",
        default=DEFAULT_CRITICAL_ROADS
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV
    )

    parser.add_argument(
        "--output-gpkg",
        default=DEFAULT_OUTPUT_GPKG
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run(
        graph_path=args.graph,
        critical_roads_path=args.critical_roads,
        output_csv=args.output_csv,
        output_gpkg=args.output_gpkg,
    )
