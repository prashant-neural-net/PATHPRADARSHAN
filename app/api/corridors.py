from fastapi import APIRouter

from ..schemas.corridor import (
    CorridorResponse,
)

from ..services.network_service import (
    get_network_graph,
)

from ..services.criticality_service import (
    get_criticality,
    get_corridors,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Corridors"],
)


@router.get(
    "/corridor",
    response_model=CorridorResponse,
)
async def get_critical_corridors():

    graph = get_network_graph()

    criticality = get_criticality(
        graph
    )

    corridors = get_corridors(
        graph,
        criticality["critical_roads"],
    )

    return {
        "corridors": corridors
    }