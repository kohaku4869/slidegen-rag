"""conftest.py — `TestClient` chạy với repo in-memory (không cần Postgres
thật), qua một `lifespan` riêng cho test (tham số `lifespan_override` của
`create_app()`)."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from slidegen_rag.api.app import create_app
from slidegen_rag.services import ingest_service
from slidegen_rag.memory.catalog_store import (
    InMemoryDocumentRepository,
    InMemoryProjectRepository,
)
from slidegen_rag.memory.memory_store import InMemorySessionRepository
from slidegen_rag.services.jobs import JobRegistry


class _StubGraph:
    """Graph giả — test chat không cần LangGraph/LLM thật (xem test_chat.py)."""

    async def ainvoke(self, state, **kwargs):
        raise NotImplementedError(
            "Stub graph — test_chat.py phải monkeypatch chat_service.run_agent_turn"
        )


@asynccontextmanager
async def _test_lifespan(app):
    document_repo = InMemoryDocumentRepository()
    session_repo = InMemorySessionRepository()
    app.state.database_url = ""
    app.state.document_repo = document_repo
    app.state.project_repo = InMemoryProjectRepository(
        document_repo=document_repo, session_repo=session_repo
    )
    app.state.session_repo = session_repo
    app.state.job_registry = JobRegistry()
    app.state.agent_lock = asyncio.Lock()
    app.state.graph = _StubGraph()
    app.state.tree_cache = OrderedDict()
    app.state.merged_ra_cache = OrderedDict()
    app.state.embedding_model = None
    yield


@pytest.fixture(autouse=True)
def _noop_ingest_submit(monkeypatch):
    """Không bao giờ thực sự chạy `run_ingest_job` (gọi model/network thật)
    trong test — các route liên quan chỉ kiểm tra hành vi tạo/đọc job."""
    monkeypatch.setattr(JobRegistry, "submit", lambda self, job, fn, *a, **kw: None)


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """`upload_document`/`start_index` ghi file `.md` thật xuống đĩa (xem
    `ingest_service.source_md_path`/`index_out_dir`) — trỏ `ROOT_DIR` sang
    thư mục tạm của test để không rác `data/projects/` thật của repo."""
    monkeypatch.setattr(ingest_service, "ROOT_DIR", tmp_path)


@pytest.fixture
def client():
    app = create_app(lifespan_override=_test_lifespan)
    with TestClient(app) as c:
        yield c
