import pandas as pd

from pipeline.failure_simulation import (
    compute_connectivity_metrics,
    find_corridor_edges,
    simulate_corridor,
)


def simulate_failures(
    graph,
    corridors,
    top_corridors=10,
    blocked_threshold=0.30,
):
    # Same ordering as the existing pipeline
    corridors = (
        corridors
        .sort_values(
            "criticality_score",
            ascending=False,
        )
        .head(top_corridors)
        .copy()
    )

    base = compute_connectivity_metrics(
        graph
    )

    rows = []

    scenarios = [
        "REMOVE",
        "CAPACITY",
        "BLOCKED",
    ]

    for _, corridor in corridors.iterrows():

        corridor_id = int(
            corridor["corridor_id"]
        )

        corridor_edges = (
            find_corridor_edges(
                graph,
                corridor,
            )
        )

        if not corridor_edges:
            continue

        for scenario in scenarios:

            (
                scenario_graph,
                affected_edges,
                metrics,
            ) = simulate_corridor(
                graph,
                corridor,
                corridor_edges,
                scenario,
                blocked_threshold,
            )

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
                        ],
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
                        ],
                    )
                )
            )

            rows.append({
                "corridor_id": corridor_id,
                "scenario": scenario,

                "corridor_segments": int(
                    corridor["segment_count"]
                ),

                "corridor_length_m": float(
                    corridor["total_length_m"]
                ),

                "corridor_betweenness": float(
                    corridor["mean_betweenness"]
                ),

                "corridor_confidence": float(
                    corridor["mean_confidence"]
                ),

                "criticality_score": float(
                    corridor["criticality_score"]
                ),

                "affected_edges": affected_edges,

                "baseline_components":
                    base["components"],

                "scenario_components":
                    metrics["components"],

                "baseline_largest_component":
                    base["largest_component"],

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

    if not rows:
        return []

    results = pd.DataFrame(rows)

    results = (
        results
        .sort_values(
            "connectivity_loss",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results.to_dict(
        orient="records"
    )