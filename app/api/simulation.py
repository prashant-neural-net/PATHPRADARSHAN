from fastapi import APIRouter

from ..schemas.simulation import (
    SimulationRequest,
)

from ..services.network_service import (
    get_network_graph,
)

from ..services.criticality_service import (
    get_criticality,
)

from ..services.criticality_service import (
    get_corridors,
)

from ..services.simulation_service import (
    simulate_failures,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Simulation"],
)


@router.post("/simulate-failure")
async def simulate_failure(
    request: SimulationRequest,
):

    graph = get_network_graph()

    criticality = get_criticality(
        graph
    )

    corridors = get_corridors(
        graph,
        criticality["critical_roads"],
        as_records=False,
    )

    results = simulate_failures(
        graph,
        corridors,
        top_corridors=request.top_corridors,
        blocked_threshold=(
            request.blocked_threshold
        ),
    )

    return {
        "results": results
    }