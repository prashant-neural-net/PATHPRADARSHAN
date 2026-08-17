from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status
)

from ..schemas.analysis import (
    AnalysisResponse
)

from ..services.analysis import (
    save_analysis_image
)


router = APIRouter(
    prefix="/api/v1/analysis",
    tags=["Analysis"]
)

@router.post(
    "",
    response_model=AnalysisResponse 
)
async def create_analysis(
    file: UploadFile = File(...)
):
    try:

        result = await save_analysis_image(
            file
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )