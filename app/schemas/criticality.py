from typing import Any

from pydantic import BaseModel


class CriticalRoad(BaseModel):
    source: int
    target: int
    edge_betweenness: float
    length_m: float
    confidence: float
    status: str
    highway: str
    osm_id: Any
    edge_id: Any


class CriticalNode(BaseModel):
    node_id: int
    x: float
    y: float
    degree: int
    degree_centrality: float
    betweenness_centrality: float
    node_type: str


class CriticalityResponse(BaseModel):
    critical_roads: list[CriticalRoad]
    critical_nodes: list[CriticalNode]
    bridge_count: int
    articulation_point_count: int