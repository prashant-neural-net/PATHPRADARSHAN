from pydantic import BaseModel


class AlternateRouteRequest(BaseModel):
    start_x: float
    start_y: float
    end_x: float
    end_y: float