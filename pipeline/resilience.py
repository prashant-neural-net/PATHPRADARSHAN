#!/usr/bin/env python3

"""
M9 - Resilience Metrics

Converts failure-simulation results into a corridor-level
resilience score.

Inputs:
    resilience_results.csv

Outputs:
    resilience_summary.csv
    resilience_corridor_ranking.csv
"""

from __future__ import annotations

import argparse

import pandas as pd


DEFAULT_INPUT = "resilience_results.csv"
DEFAULT_OUTPUT = "resilience_summary.csv"
DEFAULT_RANKING = "resilience_corridor_ranking.csv"


# ============================================================
# WEIGHTS
# ============================================================

WEIGHTS = {

    "connectivity":
        0.40,

    "largest_component":
        0.30,

    "criticality":
        0.20,

    "confidence":
        0.10,
}


# ============================================================
# RESILIENCE SCORE
# ============================================================

def calculate_score(row):

    # --------------------------------------------------------
    # Connectivity retention
    # --------------------------------------------------------

    connectivity_retention = max(
        0.0,
        min(
            1.0,
            1.0
            -
            float(
                row.get(
                    "connectivity_loss",
                    0.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Largest component retention
    # --------------------------------------------------------

    largest_component_retention = max(
        0.0,
        min(
            1.0,
            1.0
            -
            float(
                row.get(
                    "largest_component_loss",
                    0.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Criticality safety
    #
    # Higher criticality means more important corridor.
    # During failure, higher criticality means greater risk.
    #
    # Normalize using the current dataset.
    # --------------------------------------------------------

    criticality = float(
        row.get(
            "criticality_score",
            0.0
        )
    )

    criticality_safety = 1.0 / (
        1.0
        +
        criticality
    )

    # --------------------------------------------------------
    # Satellite confidence
    # --------------------------------------------------------

    confidence = max(
        0.0,
        min(
            1.0,
            float(
                row.get(
                    "corridor_confidence",
                    0.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (

        WEIGHTS["connectivity"]
        *
        connectivity_retention

        +

        WEIGHTS["largest_component"]
        *
        largest_component_retention

        +

        WEIGHTS["criticality"]
        *
        criticality_safety

        +

        WEIGHTS["confidence"]
        *
        confidence
    )

    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


# ============================================================
# PROCESS DATA
# ============================================================

def compute_resilience(
    input_path,
    output_path,
    ranking_path
):

    print(
        "Loading:",
        input_path
    )

    df = pd.read_csv(
        input_path
    )

    if df.empty:

        raise ValueError(
            f"No scenario data found in {input_path}"
        )

    print(
        "Rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required = {

        "corridor_id",

        "scenario",

        "connectivity_loss",

        "largest_component_loss",

        "corridor_confidence",

        "criticality_score",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    df = df.copy()

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    df[
        "resilience_score"
    ] = df.apply(
        calculate_score,
        axis=1
    )

    df[
        "resilience_score_100"
    ] = (
        df[
            "resilience_score"
        ]
        *
        100.0
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    df[
        "risk_score"
    ] = (
        100.0
        -
        df[
            "resilience_score_100"
        ]
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    def risk_category(score):

        if score >= 80:

            return "LOW"

        elif score >= 60:

            return "MODERATE"

        elif score >= 40:

            return "HIGH"

        else:

            return "CRITICAL"

    df[
        "risk_category"
    ] = df[
        "resilience_score_100"
    ].apply(
        risk_category
    )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # CORRIDOR-LEVEL RANKING
    # ========================================================

    ranking = (

        df.groupby(
            "corridor_id"
        )

        .agg(

            corridor_length_m=(
                "corridor_length_m",
                "first"
            ),

            corridor_segments=(
                "corridor_segments",
                "first"
            ),

            corridor_confidence=(
                "corridor_confidence",
                "first"
            ),

            criticality_score=(
                "criticality_score",
                "first"
            ),

            worst_connectivity_loss=(
                "connectivity_loss",
                "max"
            ),

            worst_component_loss=(
                "largest_component_loss",
                "max"
            ),

            minimum_resilience_score=(
                "resilience_score_100",
                "min"
            ),

            mean_resilience_score=(
                "resilience_score_100",
                "mean"
            ),
        )

        .reset_index()
    )

    # --------------------------------------------------------
    # Impact score
    # --------------------------------------------------------

    ranking[
        "impact_score"
    ] = (

        ranking[
            "worst_connectivity_loss"
        ]

        +

        ranking[
            "worst_component_loss"
        ]
    ) / 2.0

    ranking[
        "risk_score"
    ] = (
        100.0
        -
        ranking[
            "minimum_resilience_score"
        ]
    )

    ranking[
        "risk_category"
    ] = ranking[
        "minimum_resilience_score"
    ].apply(
        risk_category
    )

    ranking = ranking.sort_values(
        "risk_score",
        ascending=False
    )

    ranking.to_csv(
        ranking_path,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print(
        "=============================="
    )
    print(
        "RESILIENCE METRICS"
    )
    print(
        "=============================="
    )

    print(
        "Rows processed:",
        len(df)
    )

    print(
        "Corridors:",
        df[
            "corridor_id"
        ].nunique()
    )

    print(
        "Mean resilience:",
        round(
            df[
                "resilience_score_100"
            ].mean(),
            2
        )
    )

    print(
        "Minimum resilience:",
        round(
            df[
                "resilience_score_100"
            ].min(),
            2
        )
    )

    print()
    print(
        "TOP RISK CORRIDORS"
    )

    print(
        "------------------------------------------------------------"
    )

    for rank, (_, row) in enumerate(
        ranking.head(10).iterrows(),
        start=1
    ):

        print(

            f"{rank:2d}. "

            f"Corridor "
            f"{int(row['corridor_id'])} | "

            f"Risk: "
            f"{row['risk_score']:.2f} | "

            f"Connectivity loss: "
            f"{row['worst_connectivity_loss'] * 100:.2f}% | "

            f"Confidence: "
            f"{row['corridor_confidence']:.3f} | "

            f"Category: "
            f"{row['risk_category']}"
        )

    print()
    print(
        "Saved:",
        output_path
    )

    print(
        "Saved:",
        ranking_path
    )

    return (
        df,
        ranking
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate resilience and risk "
            "metrics from failure simulations."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--ranking",
        default=DEFAULT_RANKING
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    compute_resilience(

        input_path=
            args.input,

        output_path=
            args.output,

        ranking_path=
            args.ranking,
    )