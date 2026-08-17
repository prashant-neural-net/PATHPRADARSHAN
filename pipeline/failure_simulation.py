#!/usr/bin/env python3

"""
M8 - Corridor-level road failure simulation.

The simulation uses:
    - OSM road graph
    - Critical corridor GeoPackage

Scenarios:
    1. REMOVE   -> corridor completely removed
    2. CAPACITY -> corridor travel cost increased
    3. BLOCKED  -> corridor removed when confidence is below threshold

The implementation intentionally avoids all-pairs shortest-path
calculation because the Bengaluru graph contains 135k+ nodes.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


DEFAULT_GRAPH = "bengaluru_road_graph.gpickle"
DEFAULT_CORRIDORS = "critical_corridors.gpkg"
DEFAULT_OUTPUT = "resilience_results.csv"

DEFAULT_BLOCKED_THRESHOLD = 0.30

# Number of corridors to simulate
DEFAULT_TOP_CORRIDORS = 10


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    print("Loading graph...")

    with open(path, "rb") as f:
        G = pickle.load(f)

    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Edges:",
        G.number_of_edges()
    )

    return G


# ============================================================
# GRAPH METRICS
# ============================================================

def compute_connectivity_metrics(G):

    total_nodes = G.number_of_nodes()

    if total_nodes == 0:

        return {
            "components": 0,
            "largest_component": 0,
            "connectivity_ratio": 0.0,
            "isolated_nodes": 0,
        }

    components = list(
        nx.connected_components(G)
    )

    largest = max(
        components,
        key=len
    )

    largest_size = len(
        largest
    )

    connectivity_ratio = (
        largest_size
        /
        total_nodes
    )

    isolated_nodes = sum(
        1
        for node in G.nodes
        if G.degree(node) == 0
    )

    return {

        "components":
            len(components),

        "largest_component":
            largest_size,

        "connectivity_ratio":
            connectivity_ratio,

        "isolated_nodes":
            isolated_nodes,
    }


# ============================================================
# LOAD CORRIDORS
# ============================================================

def load_corridors(
    path,
    top_n
):

    print()
    print(
        "Loading critical corridors..."
    )

    gdf = gpd.read_file(
        path
    )

    if gdf.empty:

        raise RuntimeError(
            "Critical corridor file is empty."
        )

    required = {
        "corridor_id",
        "segment_count",
        "total_length_m",
        "mean_betweenness",
        "mean_confidence",
        "criticality_score",
    }

    missing = (
        required
        -
        set(gdf.columns)
    )

    if missing:

        raise ValueError(
            f"Missing corridor columns: {missing}"
        )

    gdf = (
        gdf
        .sort_values(
            "criticality_score",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    print(
        "Corridors selected:",
        len(gdf)
    )

    return gdf


# ============================================================
# FIND GRAPH EDGES FOR CORRIDOR
# ============================================================

def find_corridor_edges(
    G,
    corridor_row
):

    """
    Match graph edges to a corridor geometry.

    Because the corridor geometry is the union of OSM edge
    geometries, we identify graph edges that intersect/touch
    the corridor geometry.

    A small tolerance is used for numerical geometry differences.
    """

    corridor_geom = (
        corridor_row.geometry
    )

    if corridor_geom is None:

        return []

    corridor_edges = []

    # --------------------------------------------------------
    # Fast path:
    # use spatial index if available
    # --------------------------------------------------------

    edge_geometries = []

    for u, v, data in G.edges(
        data=True
    ):

        geom = data.get(
            "geometry"
        )

        if geom is None:
            continue

        edge_geometries.append(
            (
                u,
                v,
                geom
            )
        )

    # --------------------------------------------------------
    # Geometry matching
    # --------------------------------------------------------

    for u, v, geom in edge_geometries:

        try:

            if (
                geom.intersects(
                    corridor_geom
                )
            ):

                corridor_edges.append(
                    (u, v)
                )

        except Exception:

            continue

    return corridor_edges


# ============================================================
# REMOVE CORRIDOR
# ============================================================

def remove_edges(
    G,
    edges
):

    removed = 0

    for u, v in edges:

        if G.has_edge(
            u,
            v
        ):

            G.remove_edge(
                u,
                v
            )

            removed += 1

    return removed


# ============================================================
# CAPACITY FAILURE
# ============================================================

def increase_cost(
    G,
    edges,
    factor=5.0
):

    modified = 0

    for u, v in edges:

        if not G.has_edge(
            u,
            v
        ):
            continue

        old_length = float(
            G[u][v].get(
                "length_m",
                1.0
            )
        )

        G[u][v][
            "length_m"
        ] = max(
            old_length * factor,
            1.0
        )

        G[u][v][
            "capacity_factor"
        ] = factor

        modified += 1

    return modified


# ============================================================
# BASELINE
# ============================================================

def baseline_metrics(
    G
):

    metrics = (
        compute_connectivity_metrics(
            G
        )
    )

    return metrics


# ============================================================
# SIMULATE ONE CORRIDOR
# ============================================================

def simulate_corridor(
    G_base,
    corridor,
    corridor_edges,
    scenario,
    blocked_threshold
):

    G = G_base.copy()

    confidence = float(
        corridor[
            "mean_confidence"
        ]
    )

    # --------------------------------------------------------
    # REMOVE
    # --------------------------------------------------------

    if scenario == "REMOVE":

        affected_edges = (
            remove_edges(
                G,
                corridor_edges
            )
        )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    elif scenario == "CAPACITY":

        affected_edges = (
            increase_cost(
                G,
                corridor_edges,
                factor=5.0
            )
        )

    # --------------------------------------------------------
    # BLOCKED
    # --------------------------------------------------------

    elif scenario == "BLOCKED":

        if confidence < blocked_threshold:

            affected_edges = (
                remove_edges(
                    G,
                    corridor_edges
                )
            )

        else:

            affected_edges = 0

    else:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    metrics = (
        compute_connectivity_metrics(
            G
        )
    )

    return (
        G,
        affected_edges,
        metrics
    )


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(
    graph_path,
    corridors_path,
    output_path,
    top_corridors,
    blocked_threshold
):

    G_base = load_graph(
        graph_path
    )

    base = baseline_metrics(
        G_base
    )

    print()
    print("==============================")
    print("BASELINE NETWORK")
    print("==============================")

    print(
        "Components:",
        base["components"]
    )

    print(
        "Largest component:",
        base["largest_component"]
    )

    print(
        "Connectivity ratio:",
        round(
            base["connectivity_ratio"],
            4
        )
    )

    corridors = load_corridors(
        corridors_path,
        top_corridors
    )

    rows = []

    scenarios = [
        "REMOVE",
        "CAPACITY",
        "BLOCKED"
    ]

    # ========================================================
    # PROCESS CORRIDORS
    # ========================================================

    for idx, corridor in corridors.iterrows():

        corridor_id = int(
            corridor[
                "corridor_id"
            ]
        )

        print()
        print(
            "Processing corridor:",
            corridor_id
        )

        # ----------------------------------------------------
        # Find graph edges
        # ----------------------------------------------------

        corridor_edges = (
            find_corridor_edges(
                G_base,
                corridor
            )
        )

        print(
            "Matched graph edges:",
            len(corridor_edges)
        )

        if not corridor_edges:

            print(
                "WARNING: no graph edges matched."
            )

            continue

        # ----------------------------------------------------
        # Run scenarios
        # ----------------------------------------------------

        for scenario in scenarios:

            (
                G_scenario,
                affected_edges,
                metrics
            ) = simulate_corridor(

                G_base,

                corridor,

                corridor_edges,

                scenario,

                blocked_threshold
            )

            # ------------------------------------------------
            # Impact metrics
            # ------------------------------------------------

            largest_component_loss = (

                1.0
                -
                (
                    metrics[
                        "largest_component"
                    ]
                    /
                    max(
                        1,
                        base[
                            "largest_component"
                        ]
                    )
                )
            )

            connectivity_loss = (

                1.0
                -
                (
                    metrics[
                        "connectivity_ratio"
                    ]
                    /
                    max(
                        1e-9,
                        base[
                            "connectivity_ratio"
                        ]
                    )
                )
            )

            rows.append({

                "corridor_id":
                    corridor_id,

                "scenario":
                    scenario,

                "corridor_segments":
                    int(
                        corridor[
                            "segment_count"
                        ]
                    ),

                "corridor_length_m":
                    float(
                        corridor[
                            "total_length_m"
                        ]
                    ),

                "corridor_betweenness":
                    float(
                        corridor[
                            "mean_betweenness"
                        ]
                    ),

                "corridor_confidence":
                    float(
                        corridor[
                            "mean_confidence"
                        ]
                    ),

                "criticality_score":
                    float(
                        corridor[
                            "criticality_score"
                        ]
                    ),

                "affected_edges":
                    affected_edges,

                "baseline_components":
                    base[
                        "components"
                    ],

                "scenario_components":
                    metrics[
                        "components"
                    ],

                "baseline_largest_component":
                    base[
                        "largest_component"
                    ],

                "scenario_largest_component":
                    metrics[
                        "largest_component"
                    ],

                "baseline_connectivity_ratio":
                    base[
                        "connectivity_ratio"
                    ],

                "scenario_connectivity_ratio":
                    metrics[
                        "connectivity_ratio"
                    ],

                "largest_component_loss":
                    largest_component_loss,

                "connectivity_loss":
                    connectivity_loss,

                "isolated_nodes":
                    metrics[
                        "isolated_nodes"
                    ],
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    if results.empty:

        raise RuntimeError(
            "No simulation results generated."
        )

    results = (
        results
        .sort_values(
            "connectivity_loss",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    results.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("==============================")
    print("FAILURE SIMULATION COMPLETE")
    print("==============================")

    print(
        "Corridors simulated:",
        results[
            "corridor_id"
        ].nunique()
    )

    print(
        "Scenarios:",
        results[
            "scenario"
        ].nunique()
    )

    print(
        "Total simulations:",
        len(results)
    )

    print()
    print(
        "TOP NETWORK IMPACTS"
    )

    print(
        "------------------------------------------------------------"
    )

    for rank, (_, row) in enumerate(
        results.head(10).iterrows(),
        start=1
    ):

        print(

            f"{rank:2d}. "

            f"Corridor "
            f"{int(row['corridor_id'])} | "

            f"{row['scenario']} | "

            f"Connectivity loss: "
            f"{row['connectivity_loss'] * 100:.2f}% | "

            f"Largest component loss: "
            f"{row['largest_component_loss'] * 100:.2f}% | "

            f"Edges affected: "
            f"{int(row['affected_edges'])}"
        )

    print()
    print(
        "Saved:",
        output_path
    )

    return results


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Simulate critical road corridor failures."
        )
    )

    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH
    )

    parser.add_argument(
        "--corridors",
        default=DEFAULT_CORRIDORS
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--top-corridors",
        type=int,
        default=DEFAULT_TOP_CORRIDORS
    )

    parser.add_argument(
        "--blocked-threshold",
        type=float,
        default=DEFAULT_BLOCKED_THRESHOLD
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    run_simulation(

        graph_path=
            args.graph,

        corridors_path=
            args.corridors,

        output_path=
            args.output,

        top_corridors=
            args.top_corridors,

        blocked_threshold=
            args.blocked_threshold,
    )