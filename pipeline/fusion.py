#!/usr/bin/env python3
"""Fuse the V5 Sentinel-2 probability raster with OSM road geometries.

The output is an OSM edge-level GeoPackage that stores model-derived road
confidence, aggregated probability statistics, and a heuristic road status.

This script is intentionally OSM-constrained: the road geometry is treated as
structural topology while the V5 probability raster provides remote-sensing
support evidence.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString


DEFAULT_OSM = "bengaluru_large_roads_utm.gpkg"
DEFAULT_RASTER = "bengaluru_v5_probability.tif"
DEFAULT_OUTPUT = "road_edge_confidence.gpkg"
DEFAULT_SPACING = 5.0
DEFAULT_OFFSETS = (0.0, 3.0, 6.0)
DEFAULT_THRESHOLDS = {
    "open": 0.70,
    "likely_open": 0.50,
    "uncertain": 0.30,
    "likely_blocked": 0.15,
}


def iter_geometry_parts(geometry):
    """Yield LineString geometries from a geometry object."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [g for g in geometry.geoms if g.geom_type in {"LineString", "MultiLineString"}]
    return []


def safe_length(geometry):
    try:
        return float(geometry.length) if geometry is not None else 0.0
    except Exception:
        return 0.0


def point_tangent_and_normal(line, distance):
    """Return local tangent and unit normal vectors at a point along a line."""
    length = line.length
    if length <= 0:
        return (0.0, 0.0), (0.0, 0.0)

    eps = max(1e-3, min(2.0, length * 1e-4))
    start = max(0.0, distance - eps)
    stop = min(length, distance + eps)

    if abs(stop - start) < 1e-9:
        p1 = line.interpolate(max(0.0, distance - 0.1))
        p2 = line.interpolate(min(length, distance + 0.1))
    else:
        p1 = line.interpolate(start)
        p2 = line.interpolate(stop)

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    mag = math.hypot(dx, dy)
    if mag <= 1e-12:
        return (0.0, 0.0), (0.0, 0.0)

    tangent = (dx / mag, dy / mag)
    normal = (-tangent[1], tangent[0])
    return tangent, normal


def classify_confidence(confidence, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if confidence >= thresholds["open"]:
        return "OPEN"
    if confidence >= thresholds["likely_open"]:
        return "LIKELY_OPEN"
    if confidence >= thresholds["uncertain"]:
        return "UNCERTAIN"
    if confidence >= thresholds["likely_blocked"]:
        return "LIKELY_BLOCKED"
    return "BLOCKED"


def continuity_score(values, threshold=0.55):
    """Ratio of samples that remain above threshold in contiguous runs."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    valid = np.isfinite(arr)
    arr = arr[valid]
    if arr.size == 0:
        return 0.0
    high = arr >= threshold
    if high.size == 0:
        return 0.0
    longest_run = 0
    current_run = 0
    for flag in high:
        if flag:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return float(longest_run / max(1, arr.size))


def compute_confidence(values, thresholds=None):
    """Compute a bounded evidence-based score in [0, 1]."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    mean_p = float(arr.mean())
    median_p = float(np.median(arr))
    frac_055 = float((arr >= 0.55).mean())
    frac_070 = float((arr >= 0.70).mean())
    frac_080 = float((arr >= 0.80).mean())
    supported = 1.0
    if arr.size > 0:
        supported = float(arr.size / max(1, arr.size))
    cont = continuity_score(arr, threshold=0.55)

    # Weighted combination: mean, median, high-confidence evidence, continuity,
    # and support ratio. This is deliberately not pure mean probability.
    score = (
        0.35 * mean_p
        + 0.25 * median_p
        + 0.20 * frac_070
        + 0.10 * cont
        + 0.10 * supported
    )
    score = float(np.clip(score, 0.0, 1.0))
    return score


def sample_road_probability(
    geometry,
    probability_array,
    transform,
    raster_bounds,
    raster_height,
    raster_width,
    spacing=5.0,
    offsets=(0.0, 3.0, 6.0)
):
    """
    Fast road probability sampling.

    The probability raster is already loaded into RAM.
    No raster.read() calls are performed inside the sampling loop.
    """

    values = []

    if geometry is None or geometry.is_empty:
        return np.array([], dtype=np.float32), 0

    offsets = tuple(offsets)

    for line in iter_geometry_parts(geometry):

        if line is None or line.is_empty:
            continue

        length = line.length

        if length <= 0:
            continue

        steps = max(
            1,
            int(math.ceil(length / spacing))
        )

        distances = np.linspace(
            0,
            length,
            steps + 1
        )

        for distance in distances:

            p = line.interpolate(distance)

            _, normal = point_tangent_and_normal(
                line,
                distance
            )

            if normal == (0.0, 0.0):
                normal = (0.0, 1.0)

            for offset in offsets:

                # Both sides of the road
                side_offsets = (
                    (0.0,)
                    if offset == 0
                    else (offset, -offset)
                )

                for side in side_offsets:

                    x = (
                        p.x
                        + normal[0] * side
                    )

                    y = (
                        p.y
                        + normal[1] * side
                    )

                    if (
                        x < raster_bounds.left
                        or x > raster_bounds.right
                        or y < raster_bounds.bottom
                        or y > raster_bounds.top
                    ):
                        continue

                    row, col = rasterio.transform.rowcol(
                        transform,
                        x,
                        y
                    )

                    if (
                        row < 0
                        or col < 0
                        or row >= raster_height
                        or col >= raster_width
                    ):
                        continue

                    value = probability_array[
                        row,
                        col
                    ]

                    if np.isfinite(value):
                        values.append(float(value))

    arr = np.asarray(
        values,
        dtype=np.float32
    )

    return arr, int(arr.size)

def build_road_confidence_gdf(
    osm_path,
    raster_path,
    output_path,
    spacing=5.0,
    offsets=(0.0, 3.0, 6.0),
):
    print("Loading OSM roads...")

    osm_gdf = gpd.read_file(
        osm_path,
        layer="roads"
    )

    print(
        "OSM roads:",
        len(osm_gdf)
    )

    # ========================================================
    # LOAD PROBABILITY RASTER ONCE
    # ========================================================

    with rasterio.open(raster_path) as src:

        probability_array = src.read(
            1
        ).astype(np.float32)

        transform = src.transform
        bounds = src.bounds
        raster_height = src.height
        raster_width = src.width

        raster_crs = src.crs

    print(
        "Probability raster:",
        probability_array.shape
    )

    print(
        "Raster CRS:",
        raster_crs
    )

    # ========================================================
    # CRS CHECK
    # ========================================================

    if osm_gdf.crs != raster_crs:

        print(
            "Reprojecting OSM roads..."
        )

        osm_gdf = osm_gdf.to_crs(
            raster_crs
        )

    # ========================================================
    # RECORDS MUST BE CREATED BEFORE LOOP
    # ========================================================

    records = []

    total_roads = len(
        osm_gdf
    )

    # ========================================================
    # PROCESS ROADS
    # ========================================================

    for road_number, (idx, row) in enumerate(
        osm_gdf.iterrows(),
        start=1
    ):

        geom = row.geometry

        if (
            geom is None
            or geom.is_empty
        ):
            continue

        parts = iter_geometry_parts(
            geom
        )

        if not parts:
            continue

        values = []
        total_samples = 0

        # ----------------------------------------------------
        # Sample every geometry part
        # ----------------------------------------------------

        for part in parts:

            arr, count = sample_road_probability(
                part,
                probability_array,
                transform,
                bounds,
                raster_height,
                raster_width,
                spacing=spacing,
                offsets=offsets
            )

            if arr.size > 0:

                values.extend(
                    arr.tolist()
                )

            total_samples += count

        values = np.asarray(
            values,
            dtype=np.float32
        )

        valid = np.isfinite(
            values
        )

        # ====================================================
        # NO VALID SAMPLES
        # ====================================================

        if valid.sum() == 0:

            road_stats = {
                "osm_id": row.get(
                    "osm_id",
                    idx
                ),
                "highway": row.get(
                    "highway",
                    "unknown"
                ),
                "length_m": safe_length(
                    geom
                ),
                "mean_prob": 0.0,
                "median_prob": 0.0,
                "min_prob": 0.0,
                "frac_above_055": 0.0,
                "frac_above_070": 0.0,
                "frac_above_080": 0.0,
                "confidence": 0.0,
                "status": "BLOCKED",
                "sampled_points": total_samples,
                "valid_samples": 0,
            }

            records.append(
                {
                    **row.to_dict(),
                    **road_stats
                }
            )

            continue

        # ====================================================
        # VALID PROBABILITY VALUES
        # ====================================================

        values_valid = values[
            valid
        ]

        mean_prob = float(
            np.mean(values_valid)
        )

        median_prob = float(
            np.median(values_valid)
        )

        min_prob = float(
            np.min(values_valid)
        )

        frac_above_055 = float(
            np.mean(
                values_valid >= 0.55
            )
        )

        frac_above_070 = float(
            np.mean(
                values_valid >= 0.70
            )
        )

        frac_above_080 = float(
            np.mean(
                values_valid >= 0.80
            )
        )

        # ====================================================
        # CONFIDENCE SCORE
        # ====================================================

        confidence = (
            0.40 * mean_prob
            +
            0.30 * median_prob
            +
            0.20 * frac_above_055
            +
            0.10 * frac_above_070
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

        # ====================================================
        # STATUS
        # ====================================================

        if confidence >= 0.70:

            status = "OPEN"

        elif confidence >= 0.50:

            status = "LIKELY_OPEN"

        elif confidence >= 0.30:

            status = "UNCERTAIN"

        elif confidence >= 0.15:

            status = "LIKELY_BLOCKED"

        else:

            status = "BLOCKED"

        # ====================================================
        # ROAD RECORD
        # ====================================================

        road_stats = {

            "osm_id": row.get(
                "osm_id",
                idx
            ),

            "highway": row.get(
                "highway",
                "unknown"
            ),

            "length_m": safe_length(
                geom
            ),

            "mean_prob": mean_prob,

            "median_prob": median_prob,

            "min_prob": min_prob,

            "frac_above_055":
                frac_above_055,

            "frac_above_070":
                frac_above_070,

            "frac_above_080":
                frac_above_080,

            "confidence":
                confidence,

            "status":
                status,

            "sampled_points":
                total_samples,

            "valid_samples":
                int(valid.sum()),
        }

        records.append(
            {
                **row.to_dict(),
                **road_stats
            }
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        if road_number % 5000 == 0:

            print(
                f"Processed "
                f"{road_number:,} / "
                f"{total_roads:,} roads"
            )

    # ========================================================
    # CREATE OUTPUT GEODATAFRAME
    # ========================================================

    print(
        "Creating output GeoDataFrame..."
    )

    gdf = gpd.GeoDataFrame(
        records,
        geometry=[
            r.get("geometry")
            for r in records
        ],
        crs=osm_gdf.crs
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        "Saving:",
        output_path
    )

    gdf.to_file(
        output_path,
        layer="road_confidence",
        driver="GPKG"
    )

    print()
    print("==============================")
    print("OSM + V5 FUSION COMPLETE")
    print("==============================")
    print(
        "Input roads:",
        len(osm_gdf)
    )
    print(
        "Output roads:",
        len(gdf)
    )

    if len(gdf) > 0:

        print(
            "Mean confidence:",
            round(
                float(
                    gdf["confidence"].mean()
                ),
                4
            )
        )

        print(
            "\nRoad status:"
        )

        print(
            gdf["status"].value_counts()
        )

    print(
        "Saved:",
        output_path
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse V5 probability raster with OSM road geometry.")
    parser.add_argument("--osm", default=DEFAULT_OSM, help="Path to OSM road GeoPackage.")
    parser.add_argument("--raster", default=DEFAULT_RASTER, help="Path to V5 probability raster.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output GeoPackage path.")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="Sampling spacing in meters.")
    parser.add_argument("--offsets", nargs="*", type=float, default=list(DEFAULT_OFFSETS), help="Perpendicular sampling offsets in meters.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    build_road_confidence_gdf(
        osm_path=args.osm,
        raster_path=args.raster,
        output_path=args.output,
    )