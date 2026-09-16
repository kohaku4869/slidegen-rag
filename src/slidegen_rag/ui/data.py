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

import asyncio
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import numpy as np
import streamlit as st

from slidegen_rag.config import MemoryConfig, RetrieverConfig, TreeConfig
from slidegen_rag.memory.catalog import (
    DocumentRepository,
    PostgresDocumentRepository,
    PostgresProjectRepository,
    ProjectRepository,
)
from slidegen_rag.memory.dto import DocumentIndexRow, DocumentRow
from slidegen_rag.memory.engine import get_sessionmaker, ping
from slidegen_rag.memory.repository import PostgresSessionRepository, SessionRepository
from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation
from slidegen_rag.ui.merge import merge_trees
from slidegen_rag.ui.runtime import AgentRuntime

T = TypeVar("T")

# Người dùng "Local" + project "Inbox" — backfill của migration 0002, dùng
# làm chủ sở hữu mặc định (app chạy local, chưa có đăng nhập thật).
LOCAL_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
INBOX_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000002")


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


# ── Tree loading & merge (xem ui/merge.py) ──────────────────────────────────


@st.cache_resource(show_spinner=False, max_entries=8)
def load_tree_cached(path: str, mtime_ns: int):
    """`mtime_ns` trong cache key: build lại (`--force`/re-upload) đổi mtime
    -> cache tự invalidate mà không cần `.clear()` thủ công."""
    import pickle

    from slidegen_rag.tree import legacy_pickle  # noqa: F401 — cài shim unpickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _tree_builder_retriever_config():
    from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
    from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentationConfig
    from slidegen_rag.tree.retriever.tree_retriever import TreeRetrieverConfig

    embedding_model = EmbeddingModel(
        model_name=TreeConfig.EMBEDDING_MODEL_NAME,
        base_url=TreeConfig.EMBEDDING_BASE_URL or None,
    )
    tb_config = TreeBuilderConfig(
        summarization_length=TreeConfig.SUMMARIZATION_LENGTH,
        embedding_models={TreeConfig.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=TreeConfig.CLUSTER_EMBEDDING_MODEL,
    )
    tr_config = TreeRetrieverConfig(
        threshold=RetrieverConfig.THRESHOLD,
        top_k=RetrieverConfig.TOP_K,
        selection_mode=RetrieverConfig.SELECTION_MODE,
        context_embedding_model=RetrieverConfig.CONTEXT_EMBEDDING_MODEL,
        embedding_model=embedding_model,
        num_layers=RetrieverConfig.NUM_LAYERS,
        start_layer=RetrieverConfig.START_LAYER,
    )
    return RetrievalAugmentationConfig(tree_builder_config=tb_config, tree_retriever_config=tr_config)


@st.cache_resource(show_spinner=False, max_entries=2)
def _build_merged_ra_cached(parts_key: tuple[tuple[int, str, str, str], ...]) -> RetrievalAugmentation:
    """`parts_key`: tuple đã sort của `(ref_no, doc_key, title, artifact_path)`
    — dùng làm cache key thay vì `frozenset(doc_keys)` trần để build lại đúng
    khi tài liệu bị re-index (artifact_path đổi) mà không đổi tập chọn."""
    parts = []
    for ref_no, doc_key, title, artifact_path in parts_key:
        mtime_ns = Path(artifact_path).stat().st_mtime_ns
        tree = load_tree_cached(artifact_path, mtime_ns)
        parts.append((ref_no, doc_key, title, tree))
    merged_tree = merge_trees(parts)
    return RetrievalAugmentation(config=_tree_builder_retriever_config(), tree=merged_tree)


async def _load_merge_parts(
    document_repo: DocumentRepository, documents: Sequence[DocumentRow]
) -> list[tuple[int, str, str, str]]:
    indexes: list[DocumentIndexRow | None] = await asyncio.gather(
        *(document_repo.get_active_index(d.id) for d in documents)
    )
    parts = []
    for doc, idx in zip(documents, indexes):
        if idx is None or not idx.artifact_path:
            continue  # tài liệu chưa có bản index nào ready — bỏ qua, không chặn cả lượt chat
        parts.append((doc.ref_no, str(doc.id), doc.title, idx.artifact_path))
    return parts


def get_merged_ra(document_repo: DocumentRepository, documents: Sequence[DocumentRow]) -> RetrievalAugmentation | None:
    """Điểm vào DUY NHẤT để có `RetrievalAugmentation` cho một lượt chat —
    "chọn 1 tài liệu chỉ là gộp N=1" (xem plan6.md A2), nên page KHÔNG bao
    giờ tự tạo RetrievalAugmentation bằng cách khác."""
    parts_meta = run_async(_load_merge_parts(document_repo, documents))
    if not parts_meta:
        return None
    return _build_merged_ra_cached(tuple(sorted(parts_meta)))


# ── Trích dẫn: node id gộp -> DocSlice/document gốc (xem ui/merge.py, D1) ───


def build_cited_refs(document_repo: DocumentRepository, merged_tree, citations: Sequence[dict]):
    """`citations`: list `{"node_id": int, "priority": int}` (xem
    `ui/components/citations.py::extract_citations`). `node_id` là id trong
    cây GỘP hiện tại (`merged_tree`) — cần để `resolve_node` biết tài liệu
    gốc. Trả `CitedNodeRef` với `source_sha1` = sha1 của bản index ĐANG
    active tại lúc trả lời — cần để lượt sau (hoặc F5 replay) biết trích dẫn
    có còn khớp index hiện tại hay không (xem D1: "chỉ resolve chip nào có
    source_sha1 khớp index đang nạp"). Không cache — số citation mỗi lượt rất nhỏ."""
    from slidegen_rag.memory.dto import CitedNodeRef
    from slidegen_rag.ui.merge import resolve_node

    index_cache: dict[str, DocumentIndexRow | None] = {}
    refs = []
    for c in citations:
        node_id = c["node_id"]
        resolved = resolve_node(merged_tree, node_id)
        if resolved is None:
            continue
        doc_slice, _orig_idx = resolved
        if doc_slice.doc_key not in index_cache:
            index_cache[doc_slice.doc_key] = run_async(
                document_repo.get_active_index(UUID(doc_slice.doc_key))
            )
        index_row = index_cache[doc_slice.doc_key]
        refs.append(
            CitedNodeRef(
                doc_id=doc_slice.doc_id,
                source_sha1=index_row.source_sha1 if index_row else "",
                node_index=node_id,  # id trong cây GỘP — ổn định qua các lượt (offset tất định, xem A2)
                node_id=node_id,
                document_id=UUID(doc_slice.doc_key),
                priority=c.get("priority"),
            )
        )
    return refs


def source_md_path(project_id: UUID, document_id: UUID) -> Path:
    """Đường dẫn `.md` nguồn tất định theo `document_id` — KHÔNG lệ thuộc
    `doc_slug` (chỉ để hiển thị) nên hai tài liệu trùng tên file gốc trong
    cùng project không bao giờ ghi đè nhau trên đĩa."""
    from slidegen_rag.config import ROOT_DIR

    return ROOT_DIR / "data" / "projects" / str(project_id) / "src" / f"{document_id}.md"


def index_out_dir(project_id: UUID) -> Path:
    from slidegen_rag.config import ROOT_DIR

    return ROOT_DIR / "data" / "projects" / str(project_id) / "index"


def start_ingest_job(document_repo: DocumentRepository, *, project_id: UUID, document, markdown_text: str):
    """Tạo `document_indexes` row (`status='parsing'`) + đăng ký job vào
    `JobRegistry` + submit vào worker thread — xem `ui/jobs.py` (C3). Trả về
    `IngestJob` để trang lưu `job_id` vào `session_state` và poll."""
    from slidegen_rag.config import SummarizationConfig, TreeConfig
    from slidegen_rag.ui.jobs import get_job_registry, run_ingest_job

    index_row = run_async(
        document_repo.start_index(document_id=document.id, source_sha1=document.source_sha1)
    )
    invalidate_catalog()

    registry = get_job_registry()
    job = registry.new_job(
        document_id=str(document.id), index_id=str(index_row.id), doc_slug=document.doc_slug
    )
    src_path = source_md_path(project_id, document.id)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(markdown_text, encoding="utf-8")

    registry.submit(
        job,
        run_ingest_job,
        cfg_tree=TreeConfig(),
        cfg_sum=SummarizationConfig(),
        out_dir=index_out_dir(project_id),
        doc_id=str(document.id),
        source_path=str(src_path),
        markdown_text=markdown_text,
    )
    return job


def _extract_doc_summary_and_outline(tree) -> tuple[str, list[str]]:
    root = next(iter(tree.root_nodes.values()), None)
    doc_summary = root.text if root else ""
    outline_nodes = sorted(
        (n for n in tree.all_nodes.values() if n.depth == 0 and n.title),
        key=lambda n: n.line_start,
    )
    return doc_summary, [n.title for n in outline_nodes]


def finalize_ingest_job(document_repo: DocumentRepository, job) -> bool:
    """Gọi từ trang khi `job.snapshot()["state"]` là `done`/`failed`/`cancelled`
    — commit kết quả vào Postgres (`document_indexes`/`documents`, xem B3) rồi
    trả `True`. Idempotent theo nghĩa gọi 2 lần với job đã final không sao
    (finish_index/fail_index tự ghi đè cùng giá trị)."""
    import hashlib

    snap = job.snapshot()
    if snap["state"] == "done":
        pkl_path = snap["pkl_path"]
        mtime_ns = Path(pkl_path).stat().st_mtime_ns
        tree = load_tree_cached(pkl_path, mtime_ns)
        doc_summary, outline = _extract_doc_summary_and_outline(tree)
        artifact_sha1 = hashlib.sha1(Path(pkl_path).read_bytes()).hexdigest()
        stats_dict = job.stats.to_dict() if job.stats else {}
        run_async(
            document_repo.finish_index(
                UUID(job.index_id),
                artifact_path=pkl_path,
                artifact_sha1=artifact_sha1,
                build_stats=stats_dict,
                doc_summary=doc_summary,
                outline=outline,
                total_nodes=job.stats.total_nodes if job.stats else len(tree.all_nodes),
                num_layers=job.stats.num_layers if job.stats else tree.num_layers,
                total_tokens=sum(n.n_tokens for n in tree.all_nodes.values()),
            )
        )
        invalidate_catalog()
        return True
    if snap["state"] in ("failed", "cancelled"):
        run_async(
            document_repo.fail_index(
                UUID(job.index_id), error_message=snap["error"] or snap["state"]
            )
        )
        invalidate_catalog()
        return True
    return False


def get_source_excerpt(document_repo: DocumentRepository, merged_tree, node_id: int) -> str | None:
    """Đoạn `source_text` gốc (đánh số dòng theo `body_line_start/body_line_end`
    của node) — `merged_tree.source_text` rỗng có chủ đích (xem A2), phải nạp
    lại tree GỐC của đúng tài liệu qua `resolve_node`."""
    from slidegen_rag.ui.merge import resolve_node

    resolved = resolve_node(merged_tree, node_id)
    if resolved is None:
        return None
    doc_slice, _orig_idx = resolved
    index_row = run_async(document_repo.get_active_index(UUID(doc_slice.doc_key)))
    if index_row is None or not index_row.artifact_path:
        return None
    mtime_ns = Path(index_row.artifact_path).stat().st_mtime_ns
    origin_tree = load_tree_cached(index_row.artifact_path, mtime_ns)
    node = merged_tree.all_nodes.get(node_id)
    if node is None:
        return None
    lines = origin_tree.source_text.split("\n")
    return "\n".join(lines[node.body_line_start : node.body_line_end])


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
    """>3 tài liệu đang chọn -> cosine trên `doc_summary` cấp tài liệu để thu
    hẹp TRƯỚC KHI gộp cây, không lọc root khỏi cây (xem plan6.md A4). Không
    thêm LLM call — chỉ N+1 embedding call rẻ. Ảnh hưởng lên chất lượng
    top-k CHƯA ĐO ĐƯỢC — phải benchmark lại sau khi bật (xem plan6.md
    Verification #18)."""
    if len(documents) <= 3:
        return list(documents)
    model = get_embedding_model_cached()
    query_vec = np.asarray(model.create_embedding(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    scored = []
    for doc in documents:
        summary = doc.doc_summary or doc.title
        vec = np.asarray(model.create_embedding(summary), dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec = vec / vec_norm
        scored.append((float(query_vec @ vec), doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


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
    """MỘT lượt agent hoàn chỉnh dưới `runtime.agent_lock` — set_retrieval_
    augmentation -> set_web_search_enabled -> set_generate_provider ->
    seed_drill_candidates -> recorder.arun(), ĐÚNG thứ tự plan6.md A3 mô tả
    để `_ra`/`_provider`/`_web_search_enabled` (biến toàn cục ở
    tools.py/generate_node.py) không bị lượt khác ghi đè giữa chừng.

    `conv` (`ConversationManager` | None): `seed_drill_candidates()` đọc
    `describe_node()` — cũng dựa vào `_ra` toàn cục — nên PHẢI gọi sau
    `set_retrieval_augmentation(ra)`, TRONG lock; gọi trước lock (ở trang)
    sẽ đọc nhầm `_ra` của một phiên/tài liệu khác nếu có 2 lượt xếp hàng.

    Gọi qua `runtime.submit(run_agent_turn(...))` (KHÔNG `.run()`) để trang
    gọi có thể rút step khỏi `on_step` trong lúc lượt này còn chạy — xem
    plan6.md C4 và `ui/pages/chat.py`."""
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.nodes.generate_node import set_generate_provider
    from slidegen_rag.agent.observability.trajectory import TrajectoryRecorder
    from slidegen_rag.agent.tools.tools import (
        set_retrieval_augmentation,
        set_web_search_enabled,
    )

    graph = get_graph_cached()  # ngoài lock — process-wide, build 1 lần
    async with runtime.agent_lock:
        set_retrieval_augmentation(ra)
        set_web_search_enabled(web_search_enabled)
        set_generate_provider(generate_provider)
        drill_candidates = conv.seed_drill_candidates() if conv is not None else None
        recorder = TrajectoryRecorder(
            graph, tags=["ui"], metadata={"source": "streamlit"}, on_step=on_step
        )
        return await recorder.arun(
            make_initial_state(query, drill_candidates=drill_candidates),
            run_name=run_name or f"ui:{query[:60]}",
        )
