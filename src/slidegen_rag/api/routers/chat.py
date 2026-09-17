from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

from slidegen_rag.api.deps import (
    AgentLockDep,
    DocumentRepoDep,
    GraphDep,
    MergedRaCacheDep,
    SessionRepoDep,
    TreeCacheDep,
)
from slidegen_rag.api.errors import NoReadyDocumentsError, NotFoundError
from slidegen_rag.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    SourceExcerptOut,
)
from slidegen_rag.api.services import chat_service
from slidegen_rag.api.services.tree_cache import get_merged_ra

router = APIRouter(prefix="/api/v1/sessions", tags=["chat"])


@router.post("/{session_id}/chat")
async def chat(
    session_id: UUID,
    body: ChatRequest,
    request: Request,
    session_repo: SessionRepoDep,
    document_repo: DocumentRepoDep,
    graph: GraphDep,
    agent_lock: AgentLockDep,
    tree_cache: TreeCacheDep,
    ra_cache: MergedRaCacheDep,
) -> ChatResponse:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))

    result = await chat_service.ask(
        session_repo=session_repo,
        document_repo=document_repo,
        graph=graph,
        agent_lock=agent_lock,
        tree_cache=tree_cache,
        ra_cache=ra_cache,
        state=request.app.state,
        session_id=session_id,
        query=body.query,
        web_search_enabled=body.web_search_enabled,
        generate_provider=body.generate_provider,
    )
    return ChatResponse(
        answer=result.answer,
        standalone_query=result.standalone_query,
        citations=[CitationOut(**c) for c in result.citations],
    )


@router.get("/{session_id}/nodes/{node_id}/excerpt")
async def get_source_excerpt(
    session_id: UUID,
    node_id: int,
    session_repo: SessionRepoDep,
    document_repo: DocumentRepoDep,
    tree_cache: TreeCacheDep,
    ra_cache: MergedRaCacheDep,
) -> SourceExcerptOut:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    scope_doc_ids = await session_repo.get_session_scope(session_id)
    documents = [
        d for d in [await document_repo.get_document(i) for i in scope_doc_ids] if d
    ]
    ra = await get_merged_ra(
        document_repo, documents, tree_cache=tree_cache, ra_cache=ra_cache
    )
    if ra is None:
        raise NoReadyDocumentsError(str(session_id))
    excerpt = await chat_service.get_source_excerpt(
        document_repo, ra.tree, node_id, tree_cache=tree_cache
    )
    return SourceExcerptOut(node_id=node_id, excerpt=excerpt)
