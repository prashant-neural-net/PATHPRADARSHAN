from fastapi import APIRouter

from ..services.network_service import (
    get_network_graph,
    get_network_summary,
)

from ..services.criticality_service import (
    get_criticality,
    get_corridors,
)

from ..services.risk_service import (
    build_risk_ranking,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Analysis"],
)


@router.post("/analyze")
async def analyze():

    # ---------------------------------------------
    # Load graph ONCE
    # ---------------------------------------------

    graph = get_network_graph()

    # ---------------------------------------------
    # Network summary
    # ---------------------------------------------

    summary = get_network_summary(
        graph
    )

    # ---------------------------------------------
    # Criticality
    # ---------------------------------------------

    criticality = get_criticality(
        graph
    )

    # ---------------------------------------------
    # Corridors
    # ---------------------------------------------

    corridors = get_corridors(
        graph,
        criticality["critical_roads"],
        as_records=False,
    )

    # ---------------------------------------------
    # Risk ranking
    # ---------------------------------------------

    risk = build_risk_ranking(
        corridors,
        limit=10,
    )

    # ---------------------------------------------
    # Convert DataFrames to JSON
    # ---------------------------------------------

    critical_roads = (
        criticality["critical_roads"]
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    critical_nodes = (
        criticality["critical_nodes"]
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    corridor_records = (
        corridors
        .head(10)
        .drop(
            columns=["geometry"],
            errors="ignore",
        )
        .to_dict(
            orient="records"
        )
    )

    # ---------------------------------------------
    # Response
    # ---------------------------------------------

    return {

        "network_summary":
            summary,

        "critical_roads":
            critical_roads,

        "critical_nodes":
            critical_nodes,

        "critical_corridors":
            corridor_records,

        "risk_ranking":
            risk,

        "structural_vulnerability": {

            "bridge_count":
                len(
                    criticality[
                        "bridges"
                    ]
                ),

            "articulation_point_count":
                len(
                    criticality[
                        "articulation_points"
                    ]
                ),
        },
    }