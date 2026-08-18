from typing import Any

from pydantic import BaseModel


class RoadResponse(BaseModel):
    road_id: str
    source: int
    target: int
    osm_id: Any
    highway: str
    length_m: float
    mean_prob: float
    median_prob: float
    min_prob: float
    confidence: float
    status: str