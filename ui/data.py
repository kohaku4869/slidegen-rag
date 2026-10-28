"""
data.py — Facade đồng bộ DUY NHẤT giữa các trang Streamlit và tầng
async (memory/*, agent/*, tree/*) — xem plan6.md C1/C2/A3/A4.

Mọi trang gọi `run_async(...)`/các hàm `*_cached` ở đây, KHÔNG tự
`asyncio.run(...)` hay tự tạo `AgentRuntime`/repository. Import module này
là an toàn (không chạy code nặng ở top-level) — mọi model/engine chỉ khởi
tạo lười qua `st.cache_resource` khi trang thật sự cần.

**Không import `slidegen_rag.evaluate.*` ở đây** (xem `ui/app.py`).
"""

from __future__ import annotations

import importlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID

import streamlit as st

from slidegen_rag.config import INBOX_PROJECT_ID, LOCAL_USER_ID, MemoryConfig, TreeConfig  # noqa: F401
from slidegen_rag.memory.catalog import (
    DocumentRepository,
    PostgresDocumentRepository,
    PostgresProjectRepository,
    ProjectRepository,
)
from slidegen_rag.memory.dto import DocumentRow
from slidegen_rag.memory.engine import get_sessionmaker, ping
from slidegen_rag.memory.repository import PostgresSessionRepository, SessionRepository
from ui.runtime import AgentRuntime

if TYPE_CHECKING:
    from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation

T = TypeVar("T")

# Các module dưới đây kéo torch/transformers/langchain (~12s import). Import
# lười trong hàm + nạp sẵn ở thread nền (`warm_up_heavy_imports`) để trang đầu
# vẽ ngay, không trắng màn hình lúc khởi động.
_HEAVY_MODULES = (
    "slidegen_rag.services.chat_service",
    "slidegen_rag.services.ingest_service",
    "slidegen_rag.services.prefilter_service",
    "slidegen_rag.services.tree_cache",
    "slidegen_rag.services.jobs",
    "slidegen_rag.services.document_loader",
    "slidegen_rag.agent.graph.conversation",
    "slidegen_rag.agent.graph.builder",
    "slidegen_rag.tree.RetrievalAugmentation",
)


def _heavy(name: str):
    return importlib.import_module(f"slidegen_rag.services.{name}")


def get_job_registry():
    from slidegen_rag.services.jobs import get_job_registry as _get

    return _get()


@st.cache_resource(show_spinner=False)
def warm_up_heavy_imports() -> threading.Thread:
    """Khởi động một lần/process: import ngầm các module nặng trên thread
    daemon. Lượt vào trang chat đầu tiên không còn phải chờ import."""

    def _run() -> None:
        for name in _HEAVY_MODULES:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 — warm-up chỉ là tối ưu, lỗi thật sẽ lộ khi dùng
                logging.getLogger(__name__).exception("warm-up import %s lỗi", name)

    thread = threading.Thread(target=_run, name="ui-warmup", daemon=True)
    thread.start()
    return thread

# ── Runtime & repositories ──────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def get_runtime() -> AgentRuntime:
    return AgentRuntime()


def run_async(coro: Coroutine[object, object, T], timeout: float | None = None) -> T:
    """Điểm vào DUY NHẤT để chạy coroutine từ script thread của Streamlit —
    xem `ui/runtime.py`. Không bao giờ dùng `asyncio.run()` thay thế."""
    return get_runtime().run(coro, timeout=timeout)


def resolve_id_param(name: str) -> str | None:
    """Đồng bộ 2 chiều `st.query_params[name]` <-> `st.session_state[name]`.
    Ưu tiên query param (URL) vì nó sống sót qua F5 — `session_state` mất
    trắng khi trình duyệt reload (xem A1: "Refresh trình duyệt: session_state
    mới, runtime cũ"). Có giá trị chỉ ở session_state (vừa `st.switch_page`
    từ trang khác) -> ghi ngược vào query params để URL phản ánh đúng, F5
    sau đó vẫn khôi phục được (cần cho D1 Verification #13: replay sau F5)."""
    value = st.query_params.get(name)
    if value:
        st.session_state[name] = value
        return value
    value = st.session_state.get(name)
    if value:
        st.query_params[name] = value
        return value
    return None


def ensure_database_configured() -> None:
    """Gate bootstrap: app này cần Postgres cho catalog project/document —
    không có fallback in-memory khả dụng cho UI (khác CLI, xem
    `InMemorySessionRepository`), vì catalog phải sống qua nhiều lần rerun
    VÀ nhiều tab. `DATABASE_URL` rỗng -> chặn app bằng `st.stop()`, đúng
    chính sách fail-loudly của `memory/` (xem plan6.md, màn hình Settings)."""
    if not MemoryConfig.DATABASE_URL:
        st.error(
            "DATABASE_URL chưa được cấu hình. Đặt biến này trong `.env` rồi "
            "chạy `uv run alembic upgrade head` trước khi mở giao diện này."
        )
        st.stop()
    try:
        run_async(ping())
    except Exception as e:
        st.error(f"DATABASE_URL đã khai nhưng không kết nối được Postgres: {e}")
        st.stop()


async def _fetch_alembic_revision() -> str | None:
    from sqlalchemy import text

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        try:
            row = (await db.execute(text("SELECT version_num FROM alembic_version"))).first()
        except Exception:
            return None
        return row[0] if row else None


def get_alembic_revision() -> str | None:
    return run_async(_fetch_alembic_revision())


@st.cache_resource(show_spinner=False)
def get_repos() -> tuple[SessionRepository, ProjectRepository, DocumentRepository]:
    sessionmaker = get_sessionmaker()
    return (
        PostgresSessionRepository(sessionmaker),
        PostgresProjectRepository(sessionmaker),
        PostgresDocumentRepository(sessionmaker),
    )


def get_session_repo() -> SessionRepository:
    return get_repos()[0]


def get_project_repo() -> ProjectRepository:
    return get_repos()[1]


def get_document_repo() -> DocumentRepository:
    return get_repos()[2]


# ── Danh sách project/document/session — cache_data(ttl=30) ────────────────
# `.clear()` sau mỗi lần ghi (create/rename/delete/finish_index/...) — xem
# `invalidate_catalog()`.


@st.cache_data(ttl=30, show_spinner=False)
def list_projects_cached(owner_id: str):
    return run_async(get_project_repo().list_projects(owner_id=UUID(owner_id)))


@st.cache_data(ttl=30, show_spinner=False)
def list_documents_cached(project_id: str):
    return run_async(get_document_repo().list_documents(project_id=UUID(project_id)))


@st.cache_data(ttl=30, show_spinner=False)
def list_sessions_cached(project_id: str, *, include_archived: bool = False):
    return run_async(
        get_session_repo().list_sessions(
            project_id=UUID(project_id), include_archived=include_archived, limit=200
        )
    )


@st.cache_data(ttl=30, show_spinner=False)
def project_stats_cached(project_id: str):
    return run_async(get_project_repo().project_stats(UUID(project_id)))


def invalidate_catalog() -> None:
    """Gọi sau MỌI thao tác ghi vào projects/documents/sessions — cache_data
    có TTL 30s nhưng UI cần thấy thay đổi NGAY (vd. sau khi tạo project)."""
    list_projects_cached.clear()
    list_documents_cached.clear()
    list_sessions_cached.clear()
    project_stats_cached.clear()


# ── Tree loading & merge (logic ở `slidegen_rag.services.tree_cache`) ──────
# Cache LRU do `st.cache_resource` giữ để sống qua mọi rerun/tab.


@st.cache_resource(show_spinner=False)
def _tree_cache() -> OrderedDict:
    return OrderedDict()


@st.cache_resource(show_spinner=False)
def _merged_ra_cache() -> OrderedDict:
    return OrderedDict()


def load_tree_cached(path: str):
    return _heavy("tree_cache").load_tree(_tree_cache(), path)


def get_merged_ra(
    document_repo: DocumentRepository, documents: Sequence[DocumentRow]
) -> RetrievalAugmentation | None:
    """Điểm vào DUY NHẤT để có `RetrievalAugmentation` cho một lượt chat —
    "chọn 1 tài liệu chỉ là gộp N=1" (xem plan6.md A2), nên page KHÔNG bao
    giờ tự tạo RetrievalAugmentation bằng cách khác."""
    return run_async(
        _heavy("tree_cache").get_merged_ra(
            document_repo, documents, tree_cache=_tree_cache(), ra_cache=_merged_ra_cache()
        )
    )


# ── Trích dẫn & nguồn (logic ở `slidegen_rag.services.chat_service`) ────────


def build_cited_refs(document_repo: DocumentRepository, merged_tree, citations: Sequence[dict]):
    return run_async(_heavy("chat_service").abuild_cited_refs(document_repo, merged_tree, citations))


def get_source_excerpt(document_repo: DocumentRepository, merged_tree, node_id: int) -> str | None:
    return run_async(
        _heavy("chat_service").get_source_excerpt(
            document_repo, merged_tree, node_id, tree_cache=_tree_cache()
        )
    )


# ── Ingest (logic ở `slidegen_rag.services.ingest_service`) ─────────────────


def source_md_path(project_id: UUID, document_id: UUID) -> Path:
    return _heavy("ingest_service").source_md_path(project_id, document_id)


def start_ingest_job(document_repo: DocumentRepository, *, project_id: UUID, document, markdown_text: str):
    """Trả về `IngestJob` để trang lưu `job_id` vào `session_state` và poll."""
    job = run_async(
        _heavy("ingest_service").start_ingest_job(
            document_repo,
            project_id=project_id,
            document=document,
            markdown_text=markdown_text,
            job_registry=get_job_registry(),
        )
    )
    invalidate_catalog()
    return job


def finalize_ingest_job(document_repo: DocumentRepository, job) -> bool:
    done = run_async(
        _heavy("ingest_service").finalize_ingest_job(document_repo, job, tree_cache=_tree_cache())
    )
    if done:
        invalidate_catalog()
    return done


# ── A4 phase 1: tiền lọc tài liệu rẻ khi > 3 tài liệu đang chọn ─────────────


@st.cache_resource(show_spinner=False)
def get_embedding_model_cached():
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel

    return EmbeddingModel(
        model_name=TreeConfig.EMBEDDING_MODEL_NAME,
        base_url=TreeConfig.EMBEDDING_BASE_URL or None,
    )


def prefilter_documents_by_query(
    query: str, documents: Sequence[DocumentRow], top_k: int = 3
) -> list[DocumentRow]:
    """Ảnh hưởng lên chất lượng top-k CHƯA ĐO ĐƯỢC — phải benchmark lại sau
    khi bật (xem plan6.md Verification #18)."""
    if len(documents) <= 3:
        return list(documents)
    return _heavy("prefilter_service").prefilter_documents_by_query(
        get_embedding_model_cached(), query, documents, top_k
    )


# ── A3: nghiêm ngặt một lượt agent tại một thời điểm ────────────────────────


@st.cache_resource(show_spinner=False)
def get_graph_cached():
    """Compile graph MỘT LẦN cho cả process — build_graph() KHÔNG cần build
    lại khi đổi cờ web_search (retrieve_node.py đọc `web_search_enabled()`
    tại call-time, cache theo giá trị cờ — xem plan6.md A3)."""
    from slidegen_rag.agent.graph.builder import build_graph

    return build_graph()


async def run_agent_turn(
    runtime: AgentRuntime,
    ra: RetrievalAugmentation,
    *,
    query: str,
    web_search_enabled: bool,
    generate_provider: str = "gemini",
    conv=None,
    on_step=None,
    run_name: str | None = None,
):
    """Một lượt agent dưới `runtime.agent_lock` (xem
    `chat_service.run_agent_turn` về thứ tự set_* bắt buộc). Gọi qua
    `runtime.submit(run_agent_turn(...))` (KHÔNG `.run()`) để trang rút step
    khỏi `on_step` trong lúc lượt còn chạy — xem `ui/pages/chat.py`."""
    return await _heavy("chat_service").run_agent_turn(
        runtime.agent_lock,
        get_graph_cached(),
        ra,
        query=query,
        web_search_enabled=web_search_enabled,
        generate_provider=generate_provider,
        conv=conv,
        on_step=on_step,
        source="ui",
        run_name=run_name,
    )
