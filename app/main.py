from fastapi import FastAPI

from .api.health import router as health_router

from .api.analyze import router as analysis_router

from .api.network import router as network_router

from .api.corridors import router as corridor_router 

from .api.roads import router as criticality_router

from .api.simulation import router as simulation_router

from .api.alternate_route import router as alternate_route_router

from .api.risk_ranking import router as risk_ranking_router

app = FastAPI(
    title="PathPradarshan API",
    version="0.1.0"
)

app.include_router(health_router)

app.include_router(analysis_router)

app.include_router(network_router)

app.include_router(criticality_router)

app.include_router(corridor_router)

app.include_router(simulation_router)

app.include_router(alternate_route_router)

app.include_router(risk_ranking_router)