"""app.py — FastAPI app factory: lifespan (bootstrap engine/repos/graph trên
đúng event loop của uvicorn), CORS, và include routers.

Chính sách DATABASE_URL: BẮT BUỘC phải cấu hình (giống Streamlit UI, khác
CLI) — thiếu hoặc không ping được Postgres thì server fail loudly ngay lúc
khởi động, không âm thầm chạy in-memory (dữ liệu catalog phải sống qua
nhiều request/nhiều client, không có chỗ nào hợp lý để "mất khi restart").
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slidegen_rag.api.errors import register_exception_handlers
from slidegen_rag.config import MemoryConfig


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.memory.catalog import (
        PostgresDocumentRepository,
        PostgresProjectRepository,
    )
    from slidegen_rag.memory.engine import get_sessionmaker, ping
    from slidegen_rag.memory.repository import PostgresSessionRepository
    from ui.jobs import JobRegistry

    if not MemoryConfig.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL chưa được cấu hình. API server cần Postgres cho catalog "
            "project/document/session — đặt DATABASE_URL trong `.env` rồi chạy "
            "`uv run alembic upgrade head` trước khi khởi động."
        )
    await ping()  # fail loudly nếu khai báo nhưng không kết nối được

    sessionmaker = get_sessionmaker()
    app.state.database_url = MemoryConfig.DATABASE_URL
    app.state.document_repo = PostgresDocumentRepository(sessionmaker)
    app.state.project_repo = PostgresProjectRepository(sessionmaker)
    app.state.session_repo = PostgresSessionRepository(sessionmaker)

    app.state.job_registry = JobRegistry()
    app.state.agent_lock = asyncio.Lock()  # tạo trên loop hiện tại của FastAPI
    app.state.graph = build_graph()  # compile 1 lần, tái sử dụng mọi request
    app.state.tree_cache = OrderedDict()
    app.state.merged_ra_cache = OrderedDict()
    app.state.embedding_model = None

    yield


def create_app(*, lifespan_override=None) -> FastAPI:
    app = FastAPI(
        title="slidegen-rag API",
        version="0.1.0",
        lifespan=lifespan_override or lifespan,
    )

    origins = [
        o.strip()
        for o in os.getenv(
            "API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    from slidegen_rag.api.routers import (
        chat,
        documents,
        health,
        jobs,
        projects,
        sessions,
    )

    app.include_router(health.router)
    app.include_router(projects.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)

    return app


app = create_app()
