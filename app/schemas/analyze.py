from pydantic import BaseModel



class RasterMetadata(BaseModel):
    width: int
    height: int
    bands: int
    crs: str
    resolution: tuple[float, float]
    dtype: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    status: str
    raster: RasterMetadata
