"""chat_service.py — `run_agent_turn`, `abuild_cited_refs`, `get_source_excerpt`,
cộng hàm orchestration `ask()` dùng cho `POST /sessions/{id}/chat`. UI
Streamlit gọi cùng các hàm này qua `ui/data.py` (bọc `run_async`).

`agent_lock` là `asyncio.Lock` do caller sở hữu: `app.state.agent_lock` ở API,
`AgentRuntime.agent_lock` ở UI — cả hai tạo trên đúng event loop sẽ chạy
coroutine.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from slidegen_rag.agent.graph.conversation import ConversationManager
from slidegen_rag.api.errors import NoReadyDocumentsError
from slidegen_rag.services import prefilter_service
from slidegen_rag.services.tree_cache import get_merged_ra, load_tree
from slidegen_rag.memory.catalog import DocumentRepository
from slidegen_rag.memory.dto import CitedNodeRef, DocumentIndexRow, DocumentRow
from slidegen_rag.memory.repository import SessionRepository
from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation
from slidegen_rag.services.citations import extract_citations
from slidegen_rag.tree.merge import resolve_node


async def run_agent_turn(
    agent_lock: asyncio.Lock,
    graph,
    ra: RetrievalAugmentation,
    *,
    query: str,
    web_search_enabled: bool,
    generate_provider: str = "gemini",
    conv: ConversationManager | None = None,
    on_step=None,
    source: str = "api",
    run_name: str | None = None,
):
    """MỘT lượt agent hoàn chỉnh dưới `agent_lock` — set_retrieval_
    augmentation -> set_web_search_enabled -> set_generate_provider ->
    seed_drill_candidates -> recorder.arun(), ĐÚNG thứ tự để `_ra`/
    `_provider`/`_web_search_enabled` (biến toàn cục ở tools.py/
    generate_node.py) không bị lượt khác ghi đè giữa chừng.

    `on_step`: callback nhận từng `NodeStep` (UI dùng để stream tiến trình).
    `source`: nhãn tracing ("api" | "ui")."""
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.nodes.generate_node import set_generate_provider
    from slidegen_rag.agent.observability.trajectory import TrajectoryRecorder
    from slidegen_rag.agent.tools.tools import (
        set_retrieval_augmentation,
        set_web_search_enabled,
    )

    async with agent_lock:
        set_retrieval_augmentation(ra)
        set_web_search_enabled(web_search_enabled)
        set_generate_provider(generate_provider)
        drill_candidates = conv.seed_drill_candidates() if conv is not None else None
        recorder = TrajectoryRecorder(
            graph, tags=[source], metadata={"source": source}, on_step=on_step
        )
        return await recorder.arun(
            make_initial_state(query, drill_candidates=drill_candidates),
            run_name=run_name or f"{source}:{query[:60]}",
        )


async def abuild_cited_refs(
    document_repo: DocumentRepository, merged_tree, citations: Sequence[dict]
) -> list[CitedNodeRef]:
    index_cache: dict[str, DocumentIndexRow | None] = {}
    refs: list[CitedNodeRef] = []
    for c in citations:
        node_id = c["node_id"]
        resolved = resolve_node(merged_tree, node_id)
        if resolved is None:
            continue
        doc_slice, _orig_idx = resolved
        if doc_slice.doc_key not in index_cache:
            index_cache[doc_slice.doc_key] = await document_repo.get_active_index(
                UUID(doc_slice.doc_key)
            )
        index_row = index_cache[doc_slice.doc_key]
        refs.append(
            CitedNodeRef(
                doc_id=doc_slice.doc_id,
                source_sha1=index_row.source_sha1 if index_row else "",
                node_index=node_id,
                node_id=node_id,
                document_id=UUID(doc_slice.doc_key),
                priority=c.get("priority"),
            )
        )
    return refs


async def get_source_excerpt(
    document_repo: DocumentRepository,
    merged_tree,
    node_id: int,
    *,
    tree_cache: OrderedDict,
) -> str | None:
    """Đoạn `source_text` gốc — `merged_tree.source_text` rỗng có chủ đích,
    phải nạp lại tree GỐC của đúng tài liệu qua `resolve_node`."""
    resolved = resolve_node(merged_tree, node_id)
    if resolved is None:
        return None
    doc_slice, _orig_idx = resolved
    index_row = await document_repo.get_active_index(UUID(doc_slice.doc_key))
    if index_row is None or not index_row.artifact_path:
        return None
    origin_tree = load_tree(tree_cache, index_row.artifact_path)
    node = merged_tree.all_nodes.get(node_id)
    if node is None:
        return None
    lines = origin_tree.source_text.split("\n")
    return "\n".join(lines[node.body_line_start : node.body_line_end])


@dataclass
class ChatResult:
    answer: str
    standalone_query: str
    citations: list[dict]


async def ask(
    *,
    session_repo: SessionRepository,
    document_repo: DocumentRepository,
    graph,
    agent_lock: asyncio.Lock,
    tree_cache: OrderedDict,
    ra_cache: OrderedDict,
    state,
    session_id: UUID,
    query: str,
    web_search_enabled: bool,
    generate_provider: str,
) -> ChatResult:
    """Orchestration đầy đủ cho `POST /sessions/{id}/chat`: nạp scope tài
    liệu của session -> gộp cây -> resume ConversationManager -> chạy 1 lượt
    agent -> ghi lại lượt vào Postgres."""
    scope_doc_ids = await session_repo.get_session_scope(session_id)
    fetched: list[DocumentRow | None] = [
        await document_repo.get_document(d) for d in scope_doc_ids
    ]
    documents = [d for d in fetched if d is not None]
    if len(documents) > 3:
        documents = prefilter_service.prefilter_documents_by_query(
            prefilter_service.get_embedding_model(state), query, documents
        )

    ra = await get_merged_ra(
        document_repo, documents, tree_cache=tree_cache, ra_cache=ra_cache
    )
    if ra is None:
        raise NoReadyDocumentsError(str(session_id))

    conv = await ConversationManager.resume(session_repo, session_id, None)
    standalone_query = await conv.contextualize(query)

    final_state = await run_agent_turn(
        agent_lock,
        graph,
        ra,
        query=standalone_query,
        web_search_enabled=web_search_enabled,
        generate_provider=generate_provider,
        conv=conv,
        run_name=f"api:{query[:60]}",
    )

    citations = extract_citations(final_state)
    cited_refs = await abuild_cited_refs(document_repo, ra.tree, citations)
    await conv.arecord_turn(
        query,
        standalone_query,
        final_state["generation"],
        final_state["document_node_ids"],
        cited_refs=cited_refs,
    )
    await conv.maybe_compact()

    return ChatResult(
        answer=final_state["generation"],
        standalone_query=standalone_query,
        citations=citations,
    )
