from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from ..schemas.network import NetworkSummary
from ..services.network_service import (
    get_network_summary,
)


router = APIRouter(
    prefix="/api/v1/network",
    tags=["Network"],
)


@router.get(
    "/summary",
    response_model=NetworkSummary,
)
async def network_summary():

    try:

        return get_network_summary()

    except FileNotFoundError:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Network data has not "
                "been generated yet."
            ),
        )