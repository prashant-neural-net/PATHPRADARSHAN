from pydantic import BaseModel


class NetworkSummary(BaseModel):
    total_junctions: int
    total_roads: int
    total_road_length_m: float
    total_road_length_km: float