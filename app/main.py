from fastapi import FastAPI

from .api.health import router as health_router

from .api.analysis import router as analysis_router

app = FastAPI(
    title="PathPradarshan API",
    version="0.1.0"
)

app.include_router(health_router)

app.include_router(analysis_router)
