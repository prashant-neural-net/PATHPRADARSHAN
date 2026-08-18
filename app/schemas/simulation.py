from pydantic import BaseModel


class SimulationRequest(BaseModel):
    top_corridors: int = 10
    blocked_threshold: float = 0.30