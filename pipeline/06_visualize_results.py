#!/usr/bin/env python3

"""
M10 - Visualization

Creates:
1. OSM roads by V5 confidence/status
2. Top critical roads
3. Critical corridor map
4. Corridor risk ranking
5. Example critical-corridor failure map
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CONFIDENCE = "road_edge_confidence.gpkg"

DEFAULT_CRITICAL_ROADS = "critical_roads.csv"

DEFAULT_CRITICAL_GPKG = "top_critical_roads.gpkg"

DEFAULT_CORRIDORS = "critical_corridors.gpkg"

DEFAULT_RANKING = "resilience_corridor_ranking.csv"

DEFAULT_GRAPH = "bengaluru_road_graph.gpickle"

DEFAULT_OUTPUT_DIR = "visualizations"


STATUS_COLORS = {

    "OPEN": "#2ca02c",

    "LIKELY_OPEN": "#98df8a",

    "UNCERTAIN": "#ffbb78",

    "LIKELY_BLOCKED": "#ff7f0e",

    "BLOCKED": "#d62728",
}


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    with open(path, "rb") as handle:

        return pickle.load(handle)


# ============================================================
# 1. CONFIDENCE MAP
# ============================================================

def plot_confidence_map(
    gdf,
    output_path
):

    print(
        "Creating confidence map..."
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    if gdf.empty:

        ax.text(
            0.5,
            0.5,
            "No roads to display",
            ha="center",
            va="center"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        return

    for status, color in STATUS_COLORS.items():

        if "status" not in gdf.columns:
            continue

        subset = gdf[
            gdf["status"] == status
        ]

        if not subset.empty:

            subset.plot(
                ax=ax,
                color=color,
                linewidth=0.7,
                label=status
            )

    ax.set_title(
        "Bengaluru OSM Roads — V5 Satellite Evidence",
        fontsize=15
    )

    ax.legend(
        frameon=False
    )

    ax.set_axis_off()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 2. CRITICAL ROADS
# ============================================================

def plot_critical_roads(
    critical_gpkg,
    output_path
):

    print(
        "Creating critical-road map..."
    )

    gdf = gpd.read_file(
        critical_gpkg
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    if gdf.empty:

        ax.text(
            0.5,
            0.5,
            "No critical roads",
            ha="center",
            va="center"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        return

    gdf.plot(
        ax=ax,
        column="edge_betweenness",
        cmap="viridis",
        linewidth=2.0,
        legend=True
    )

    ax.set_title(
        "Top Critical Roads — Edge Betweenness",
        fontsize=15
    )

    ax.set_axis_off()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 3. CRITICAL CORRIDORS
# ============================================================

def plot_critical_corridors(
    corridor_path,
    output_path
):

    print(
        "Creating critical-corridor map..."
    )

    gdf = gpd.read_file(
        corridor_path
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    if gdf.empty:

        ax.text(
            0.5,
            0.5,
            "No critical corridors",
            ha="center",
            va="center"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        return

    gdf.plot(
        ax=ax,
        column="criticality_score",
        cmap="plasma",
        linewidth=2.5,
        legend=True
    )

    # --------------------------------------------------------
    # Corridor labels
    # --------------------------------------------------------

    for _, row in gdf.iterrows():

        try:

            point = row.geometry.representative_point()

            corridor_id = int(
                row["corridor_id"]
            )

            ax.annotate(

                f"C{corridor_id}",

                (
                    point.x,
                    point.y
                ),

                fontsize=7
            )

        except Exception:

            continue

    ax.set_title(
        "Critical Road Corridors",
        fontsize=15
    )

    ax.set_axis_off()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 4. RISK RANKING
# ============================================================

def plot_risk_ranking(
    ranking_path,
    output_path
):

    print(
        "Creating risk-ranking chart..."
    )

    df = pd.read_csv(
        ranking_path
    )

    if df.empty:

        return

    df = (
        df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
        .copy()
    )

    labels = [
        f"C{int(x)}"
        for x in df[
            "corridor_id"
        ]
    ]

    values = df[
        "risk_score"
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Top Critical Corridors — Risk Ranking",
        fontsize=15
    )

    ax.set_xlabel(
        "Corridor"
    )

    ax.set_ylabel(
        "Risk Score"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 5. FAILURE SCENARIO
# ============================================================

def plot_corridor_failure(
    graph_path,
    corridor_path,
    ranking_path,
    output_path
):

    print(
        "Creating corridor-failure map..."
    )

    G = load_graph(
        graph_path
    )

    corridors = gpd.read_file(
        corridor_path
    )

    ranking = pd.read_csv(
        ranking_path
    )

    if corridors.empty:
        return

    if ranking.empty:
        return

    # --------------------------------------------------------
    # Select highest-risk corridor
    # --------------------------------------------------------

    ranking = ranking.sort_values(
        "risk_score",
        ascending=False
    )

    top_corridor_id = int(
        ranking.iloc[0]["corridor_id"]
    )

    corridor = corridors[
        corridors["corridor_id"]
        ==
        top_corridor_id
    ]

    if corridor.empty:
        return

    # --------------------------------------------------------
    # Convert graph edges to GeoDataFrame
    # --------------------------------------------------------

    print(
        "Preparing graph geometries..."
    )

    features = []

    for u, v, data in G.edges(
        data=True
    ):

        geom = data.get(
            "geometry"
        )

        if geom is None:
            continue

        features.append({
            "geometry": geom
        })

    print(
        "Graph geometries:",
        len(features)
    )

    if not features:
        return

    roads = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=corridors.crs
    )

    # --------------------------------------------------------
    # Plot everything in bulk
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    print(
        "Plotting base road network..."
    )

    roads.plot(
        ax=ax,
        color="#d9d9d9",
        linewidth=0.25
    )

    print(
        "Plotting failed corridor..."
    )

    corridor.plot(
        ax=ax,
        color="#d62728",
        linewidth=4.0,
        label=(
            f"Failed Corridor {top_corridor_id}"
        )
    )

    # --------------------------------------------------------
    # Add information box
    # --------------------------------------------------------

    top_row = ranking.iloc[0]

    info = (
        f"Corridor {top_corridor_id}\n"
        f"Risk score: "
        f"{top_row['risk_score']:.2f}\n"
        f"Connectivity impact: "
        f"{top_row['worst_connectivity_loss'] * 100:.2f}%"
    )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    ax.set_title(
        (
            f"Critical Corridor Failure — "
            f"Corridor {top_corridor_id}"
        ),
        fontsize=15
    )

    ax.legend(
        frameon=False
    )

    ax.set_axis_off()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path
    )

# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create visualizations for the "
            "OSM-constrained road resilience pipeline."
        )
    )

    parser.add_argument(
        "--confidence",
        default=DEFAULT_CONFIDENCE
    )

    parser.add_argument(
        "--critical-gpkg",
        default=DEFAULT_CRITICAL_GPKG
    )

    parser.add_argument(
        "--corridors",
        default=DEFAULT_CORRIDORS
    )

    parser.add_argument(
        "--ranking",
        default=DEFAULT_RANKING
    )

    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Confidence
    # --------------------------------------------------------

    confidence_gdf = gpd.read_file(
        args.confidence
    )

    plot_confidence_map(
        confidence_gdf,
        out_dir /
        "osm_roads_confidence.png"
    )

    # --------------------------------------------------------
    # 2. Critical roads
    # --------------------------------------------------------

    if Path(
        args.critical_gpkg
    ).exists():

        plot_critical_roads(
            args.critical_gpkg,
            out_dir /
            "top_critical_roads.png"
        )

    # --------------------------------------------------------
    # 3. Critical corridors
    # --------------------------------------------------------

    if Path(
        args.corridors
    ).exists():

        plot_critical_corridors(
            args.corridors,
            out_dir /
            "critical_corridors.png"
        )

    # --------------------------------------------------------
    # 4. Risk ranking
    # --------------------------------------------------------

    if Path(
        args.ranking
    ).exists():

        plot_risk_ranking(
            args.ranking,
            out_dir /
            "corridor_risk_ranking.png"
        )

    # --------------------------------------------------------
    # 5. Failure scenario
    # --------------------------------------------------------

    if (
        Path(args.graph).exists()
        and
        Path(args.corridors).exists()
        and
        Path(args.ranking).exists()
    ):

        plot_corridor_failure(
            args.graph,
            args.corridors,
            args.ranking,
            out_dir /
            "critical_corridor_failure.png"
        )

    print()
    print(
        "=============================="
    )

    print(
        "VISUALIZATION COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        "Saved visualizations to:",
        out_dir
    )