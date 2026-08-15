#!/usr/bin/env python3

"""
Build an OSM-backed road graph using V5 satellite confidence.

Architecture:

OSM road centerlines
        ↓
Spatial index
        ↓
Road-road intersections
        ↓
Split roads at intersections
        ↓
NetworkX graph
        ↓
Confidence/status stored on edges
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np

from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    GeometryCollection,
)

from shapely.ops import split, substring


DEFAULT_INPUT = "road_edge_confidence.gpkg"
DEFAULT_OUTPUT = "bengaluru_road_graph.gpickle"


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def iter_line_parts(geometry):

    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":

        parts = []

        for g in geometry.geoms:

            if g.geom_type == "LineString":
                parts.append(g)

            elif g.geom_type == "MultiLineString":
                parts.extend(list(g.geoms))

        return parts

    return []


def extract_intersection_points(geometry):

    if geometry is None or geometry.is_empty:
        return []

    geom_type = geometry.geom_type

    if geom_type == "Point":
        return [geometry]

    if geom_type == "MultiPoint":
        return list(geometry.geoms)

    if geom_type == "GeometryCollection":

        points = []

        for g in geometry.geoms:

            if g.geom_type == "Point":
                points.append(g)

            elif g.geom_type == "MultiPoint":
                points.extend(list(g.geoms))

        return points

    # LineString intersection means overlapping roads.
    # We intentionally don't create infinite intersection points.
    return []


def point_key(point, precision=6):

    return (
        round(float(point.x), precision),
        round(float(point.y), precision),
    )


# ============================================================
# SEGMENT PREPARATION
# ============================================================

def make_road_segments(gdf):

    segments = []

    for idx, row in gdf.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        parts = iter_line_parts(
            geometry
        )

        for part_idx, part in enumerate(parts):

            if (
                part is None
                or part.is_empty
                or part.length <= 0
            ):
                continue

            attr = row.to_dict()

            attr["source_row_id"] = idx
            attr["part_idx"] = part_idx

            attr["edge_id"] = (
                f"{row.get('osm_id', idx)}_{part_idx}"
            )

            segments.append(
                {
                    "geometry": part,
                    **attr,
                }
            )

    return segments


# ============================================================
# EDGE ATTRIBUTES
# ============================================================

def edge_attributes(seg, geometry):

    return {

        "osm_id":
            seg.get("osm_id"),

        "highway":
            seg.get("highway", "unknown"),

        "length_m":
            float(
                geometry.length
            ),

        "mean_prob":
            float(
                seg.get(
                    "mean_prob",
                    0.0
                )
            ),

        "median_prob":
            float(
                seg.get(
                    "median_prob",
                    0.0
                )
            ),

        "min_prob":
            float(
                seg.get(
                    "min_prob",
                    0.0
                )
            ),

        "frac_above_055":
            float(
                seg.get(
                    "frac_above_055",
                    0.0
                )
            ),

        "frac_above_070":
            float(
                seg.get(
                    "frac_above_070",
                    0.0
                )
            ),

        "frac_above_080":
            float(
                seg.get(
                    "frac_above_080",
                    0.0
                )
            ),

        "confidence":
            float(
                seg.get(
                    "confidence",
                    0.0
                )
            ),

        "status":
            seg.get(
                "status",
                "UNCERTAIN"
            ),

        "source_row_id":
            seg.get(
                "source_row_id"
            ),

        "part_idx":
            seg.get(
                "part_idx"
            ),

        "edge_id":
            seg.get(
                "edge_id"
            ),

        "geometry":
            geometry,
    }


# ============================================================
# ADD NODE
# ============================================================

def add_node(
    G,
    node_map,
    point,
    node_type="junction",
):

    key = point_key(
        point
    )

    if key not in node_map:

        node_id = len(
            node_map
        )

        node_map[key] = node_id

        G.add_node(
            node_id,
            x=key[0],
            y=key[1],
            node_type=node_type,
        )

    return node_map[key]


# ============================================================
# BUILD GRAPH
# ============================================================

def build_osm_graph(gdf):

    print()
    print("==============================")
    print("PREPARING OSM GRAPH")
    print("==============================")

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    gdf = gdf.copy()

    gdf = gdf[
        gdf.geometry.notnull()
    ].copy()

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    gdf = gdf[
        gdf.geom_type.isin(
            {
                "LineString",
                "MultiLineString",
            }
        )
    ].copy()

    print(
        "Input road features:",
        len(gdf)
    )

    # --------------------------------------------------------
    # Convert MultiLineStrings to individual segments
    # --------------------------------------------------------

    segments = make_road_segments(
        gdf
    )

    print(
        "Line segments:",
        len(segments)
    )

    geometries = [
        seg["geometry"]
        for seg in segments
    ]

    # --------------------------------------------------------
    # Spatial index
    # --------------------------------------------------------

    print(
        "Building spatial index..."
    )

    road_tree = gpd.GeoSeries(
        geometries,
        crs=gdf.crs
    ).sindex

    # --------------------------------------------------------
    # Store intersection points PER road
    # --------------------------------------------------------

    intersection_points = [
        []
        for _ in segments
    ]

    intersection_count = 0

    print(
        "Detecting road intersections..."
    )

    # ========================================================
    # FIND INTERSECTIONS
    # ========================================================

    for i, geom_i in enumerate(
        geometries
    ):

        candidate_ids = road_tree.query(
            geom_i,
            predicate="intersects"
        )

        for raw_j in candidate_ids:

            j = int(raw_j)

            # Only process pair once
            if j <= i:
                continue

            geom_j = geometries[j]

            if geom_i.equals(
                geom_j
            ):
                continue

            intersection = (
                geom_i.intersection(
                    geom_j
                )
            )

            points = extract_intersection_points(
                intersection
            )

            if not points:
                continue

            for point in points:

                # Ignore intersections that are
                # effectively endpoints.
                d_i = min(
                    point.distance(
                        Point(
                            geom_i.coords[0]
                        )
                    ),
                    point.distance(
                        Point(
                            geom_i.coords[-1]
                        )
                    ),
                )

                d_j = min(
                    point.distance(
                        Point(
                            geom_j.coords[0]
                        )
                    ),
                    point.distance(
                        Point(
                            geom_j.coords[-1]
                        )
                    ),
                )

                # Store only genuine interior
                # intersection points.
                if d_i > 0.01:

                    intersection_points[
                        i
                    ].append(point)

                if d_j > 0.01:

                    intersection_points[
                        j
                    ].append(point)

                intersection_count += 1

        if (
            (i + 1) % 5000 == 0
        ):

            print(
                f"Intersection scan: "
                f"{i + 1:,} / "
                f"{len(segments):,}"
            )

    print(
        "Raw intersection points:",
        intersection_count
    )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    print()
    print(
        "Building topology..."
    )

    G = nx.Graph()

    node_map = {}

    edge_count = 0

    split_road_count = 0

    # ========================================================
    # SPLIT EACH ROAD AT INTERSECTIONS
    # ========================================================

    for index, seg in enumerate(
        segments
    ):

        line = seg[
            "geometry"
        ]

        points = intersection_points[
            index
        ]

        # ----------------------------------------------------
        # Remove duplicate points
        # ----------------------------------------------------

        unique = {}

        for point in points:

            key = point_key(
                point
            )

            unique[key] = point

        points = list(
            unique.values()
        )

        # ----------------------------------------------------
        # Create endpoints
        # ----------------------------------------------------

        start = Point(
            line.coords[0]
        )

        end = Point(
            line.coords[-1]
        )

        all_points = [
            start,
            *points,
            end,
        ]

        # ----------------------------------------------------
        # Sort points along road
        # ----------------------------------------------------

        all_points.sort(
            key=lambda p:
                line.project(p)
        )

        # ----------------------------------------------------
        # Remove duplicates after sorting
        # ----------------------------------------------------

        ordered = []

        seen = set()

        for point in all_points:

            key = point_key(
                point
            )

            if key in seen:
                continue

            seen.add(key)
            ordered.append(point)

        # ----------------------------------------------------
        # Create road sub-edges
        # ----------------------------------------------------

        if len(ordered) > 2:

            split_road_count += 1

        for a, b in zip(
            ordered[:-1],
            ordered[1:]
        ):

            distance_a = line.project(
                a
            )

            distance_b = line.project(
                b
            )

            if (
                distance_b
                <= distance_a
            ):
                continue

            # Extract actual geometry
            piece = substring(
                line,
                distance_a,
                distance_b
            )

            if (
                piece is None
                or piece.is_empty
                or piece.length <= 0
            ):
                continue

            node_a = add_node(
                G,
                node_map,
                a,
                node_type=(
                    "intersection"
                    if a in points
                    else "endpoint"
                ),
            )

            node_b = add_node(
                G,
                node_map,
                b,
                node_type=(
                    "intersection"
                    if b in points
                    else "endpoint"
                ),
            )

            if node_a == node_b:
                continue

            attrs = edge_attributes(
                seg,
                piece
            )

            # ------------------------------------------------
            # Graph uses node pair as topology.
            # Preserve strongest confidence if duplicate
            # connection appears.
            # ------------------------------------------------

            if G.has_edge(node_a, node_b):

                existing = G.get_edge_data(
                    node_a,
                    node_b
                )

                if (
                    attrs["confidence"]
                    >
                    existing.get(
                        "confidence",
                        0.0
                    )
                ):
                    G.add_edge(
                        node_a,
                        node_b,
                        **attrs
                    )

            else:

                G.add_edge(
                    node_a,
                    node_b,
                    **attrs
                )

                edge_count += 1

        if (
            (index + 1) % 5000 == 0
        ):

            print(
                f"Topology: "
                f"{index + 1:,} / "
                f"{len(segments):,}"
            )

    # ========================================================
    # REMOVE ISOLATED NODES
    # ========================================================

    isolated = [
        node
        for node, degree
        in G.degree()
        if degree == 0
    ]

    G.remove_nodes_from(
        isolated
    )

    # ========================================================
    # GRAPH STATISTICS
    # ========================================================

    components = list(
        nx.connected_components(
            G
        )
    )

    largest_component = max(
        (
            len(c)
            for c in components
        ),
        default=0
    )

    print()
    print("==============================")
    print("OSM GRAPH BUILD COMPLETE")
    print("==============================")

    print(
        "Segments used:",
        len(segments)
    )

    print(
        "Roads split at intersections:",
        split_road_count
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
        "Connected components:",
        len(components)
    )

    print(
        "Largest connected component:",
        largest_component
    )

    if G.number_of_nodes() > 0:

        largest_percent = (
            largest_component
            /
            G.number_of_nodes()
        ) * 100

        print(
            "Largest component %:",
            round(
                largest_percent,
                2
            )
        )

    return G


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build an OSM-based road graph "
            "with V5 confidence attributes."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Road confidence GeoPackage",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output NetworkX graph",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    print(
        "Loading:",
        args.input
    )

    gdf = gpd.read_file(
        args.input
    )

    print(
        "CRS:",
        gdf.crs
    )

    graph = build_osm_graph(
        gdf
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "wb"
    ) as handle:

        pickle.dump(
            graph,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print()
    print(
        "Saved graph:",
        output_path
    )