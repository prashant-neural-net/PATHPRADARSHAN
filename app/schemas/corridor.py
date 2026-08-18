from typing import Any

from pydantic import BaseModel


class Corridor(BaseModel):
    corridor_id: int
    segment_count: int
    total_length_m: float
    mean_betweenness: float
    max_betweenness: float
    mean_confidence: float
    min_confidence: float
    dominant_status: str
    dominant_highway: str
    criticality_score: float
    osm_segment_count: int


class CorridorResponse(BaseModel):
    corridors: list[Corridor]