from fastapi import (
    APIRouter,
    HTTPException,
)

from ..schemas.criticality import (
    CriticalityResponse,
)

from ..schemas.road import (
    RoadResponse
)

from ..services.criticality_service import (
    get_criticality_response,
)

from ..services.network_service import (
    get_network_graph,
)

from ..services.road_service import (
    get_road
)


router = APIRouter(
    prefix="/api/v1/roads",
    tags=["Roads"],
)


from fastapi import HTTPException


@router.get(
    "/roads/{road_id}",
    response_model=RoadResponse,
)
async def get_road_details(
    road_id: str,
):

    road = get_road(
        road_id
    )

    if road is None:

        raise HTTPException(
            status_code=404,
            detail="Road not found",
        )

    return road

@router.get(
    "/critical",
    response_model=CriticalityResponse,
)
async def get_critical_roads():

    try:

        graph = get_network_graph()

        return get_criticality_response(
            graph
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )