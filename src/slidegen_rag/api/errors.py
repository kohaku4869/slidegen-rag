"""errors.py — Domain exceptions dùng xuyên suốt `api/routers/` và
`api/services/`, cộng các exception handler map chúng sang HTTP status.

Route handler raise các exception này thay vì tự trả `HTTPException` rải
rác — giữ logic domain (services/) không phụ thuộc FastAPI.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError


class NotFoundError(Exception):
    def __init__(self, resource: str, resource_id: str):
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} không tồn tại: {resource_id}")


class ConflictError(Exception):
    pass


class DatabaseUnavailableError(Exception):
    pass


class NoReadyDocumentsError(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} chưa có tài liệu nào index xong trong scope"
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": f"{exc.resource} không tồn tại: {exc.resource_id}"},
        )

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NoReadyDocumentsError)
    async def _no_ready_documents(
        _: Request, exc: NoReadyDocumentsError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DatabaseUnavailableError)
    async def _db_unavailable(
        _: Request, exc: DatabaseUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": str(exc) or "Database unavailable"}
        )

    @app.exception_handler(SQLAlchemyError)
    async def _sqlalchemy_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": f"Database unavailable: {exc}"}
        )
