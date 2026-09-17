"""deps.py — `Depends()` providers đọc tài nguyên từ `request.app.state`.

Mọi provider đọc từ `request.app.state` (không phải biến module-global) để
test có thể set state riêng cho từng `TestClient`/`lifespan` (xem
`tests/unit/api/conftest.py`).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, Request

from slidegen_rag.memory.catalog import DocumentRepository, ProjectRepository
from slidegen_rag.memory.repository import SessionRepository
from ui.jobs import JobRegistry


def get_document_repo(request: Request) -> DocumentRepository:
    return request.app.state.document_repo


def get_project_repo(request: Request) -> ProjectRepository:
    return request.app.state.project_repo


def get_session_repo(request: Request) -> SessionRepository:
    return request.app.state.session_repo


def get_job_registry(request: Request) -> JobRegistry:
    return request.app.state.job_registry


def get_agent_lock(request: Request) -> asyncio.Lock:
    return request.app.state.agent_lock


def get_graph(request: Request):
    return request.app.state.graph


def get_tree_cache(request: Request) -> dict:
    return request.app.state.tree_cache


def get_merged_ra_cache(request: Request) -> dict:
    return request.app.state.merged_ra_cache


DocumentRepoDep = Annotated[DocumentRepository, Depends(get_document_repo)]
ProjectRepoDep = Annotated[ProjectRepository, Depends(get_project_repo)]
SessionRepoDep = Annotated[SessionRepository, Depends(get_session_repo)]
JobRegistryDep = Annotated[JobRegistry, Depends(get_job_registry)]
AgentLockDep = Annotated[asyncio.Lock, Depends(get_agent_lock)]
GraphDep = Annotated[object, Depends(get_graph)]
TreeCacheDep = Annotated[dict, Depends(get_tree_cache)]
MergedRaCacheDep = Annotated[dict, Depends(get_merged_ra_cache)]
