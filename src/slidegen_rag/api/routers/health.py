from __future__ import annotations

from fastapi import APIRouter, Request

from slidegen_rag.api.errors import DatabaseUnavailableError
from slidegen_rag.memory.engine import ping

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/db")
async def health_db(request: Request) -> dict:
    database_url = getattr(request.app.state, "database_url", None)
    if not database_url:
        raise DatabaseUnavailableError("DATABASE_URL chưa được cấu hình")
    try:
        await ping(database_url)
    except Exception as exc:
        raise DatabaseUnavailableError(str(exc)) from exc
    return {"status": "ok"}
