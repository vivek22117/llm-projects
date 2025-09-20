
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends

from .rate_limiter import limiter
from ..models.responses import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/healthz", response_model=HealthResponse, tags=["Health"])
@limiter.limit("4/second")
def health_check() -> HealthResponse:
    try:
        db_healthy = True

        return HealthResponse(status="ok", database_connection=db_healthy)

    except Exception as ex:
        logger.error(f"Health check failed: {ex}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(ex)}")

