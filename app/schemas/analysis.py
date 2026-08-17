from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    status: str
