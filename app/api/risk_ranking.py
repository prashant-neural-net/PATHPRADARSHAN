from fastapi import APIRouter

from ..services.network_service import (
    get_network_graph,
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
    tags=["Risk"],
)


@router.get("/risk-ranking")
async def risk_ranking():

    graph = get_network_graph()

    criticality = get_criticality(
        graph
    )

    corridors = get_corridors(
        graph,
        criticality["critical_roads"],
        as_records=False,
    )

    ranking = build_risk_ranking(
        corridors
    )

    return {
        "count": len(ranking),
        "ranking": ranking,
    }