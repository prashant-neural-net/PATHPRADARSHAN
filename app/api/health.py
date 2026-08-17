from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
def get_health():
    return {"status": "healthy", "service": "route-resilience-api"}
