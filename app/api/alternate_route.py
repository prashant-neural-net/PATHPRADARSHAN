from fastapi import APIRouter, HTTPException

from ..schemas.routing import AlternateRouteRequest
from ..services.network_service import get_network_graph
from ..services.routing_service import find_route


router = APIRouter(
    prefix="/api/v1",
    tags=["Routing"],
)


@router.post("/alternate-route")
async def alternate_route(
    request: AlternateRouteRequest,
):
    graph = get_network_graph()

    try:
        route = find_route(
            graph,
            start_x=request.start_x,
            start_y=request.start_y,
            end_x=request.end_x,
            end_y=request.end_y,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return route