from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from slidegen_rag.api.deps import ProjectRepoDep, SessionRepoDep
from slidegen_rag.api.errors import NotFoundError
from slidegen_rag.api.schemas.sessions import (
    MessageOut,
    SessionCreate,
    SessionDetailOut,
    SessionListOut,
    SessionRename,
    SessionScopeIn,
)
from slidegen_rag.config import MemoryConfig
from ui.data import LOCAL_USER_ID

router = APIRouter(prefix="/api/v1", tags=["sessions"])


@router.post("/projects/{project_id}/sessions", status_code=201)
async def create_session(
    project_id: UUID,
    body: SessionCreate,
    project_repo: ProjectRepoDep,
    session_repo: SessionRepoDep,
) -> SessionDetailOut:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    state = await session_repo.create_session(
        title=body.title,
        project_id=project_id,
        document_id=body.document_id,
        user_id=LOCAL_USER_ID,
        max_turns=MemoryConfig.SESSION_MAX_TURNS,
    )
    scope = await session_repo.get_session_scope(state.id)
    return SessionDetailOut(
        id=state.id,
        title=state.title,
        bound_doc_id=state.bound_doc_id,
        max_turns=state.max_turns,
        history_summary=state.history_summary,
        scope_document_ids=scope,
    )


@router.get("/projects/{project_id}/sessions")
async def list_sessions(
    project_id: UUID,
    project_repo: ProjectRepoDep,
    session_repo: SessionRepoDep,
    include_archived: bool = False,
) -> list[SessionListOut]:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    rows = await session_repo.list_sessions(
        project_id=project_id, include_archived=include_archived, limit=200
    )
    return [SessionListOut.from_row(r) for r in rows]


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: UUID, session_repo: SessionRepoDep
) -> SessionDetailOut:
    state = await session_repo.load_session(session_id)
    if state is None:
        raise NotFoundError("session", str(session_id))
    scope = await session_repo.get_session_scope(session_id)
    return SessionDetailOut(
        id=state.id,
        title=state.title,
        bound_doc_id=state.bound_doc_id,
        max_turns=state.max_turns,
        history_summary=state.history_summary,
        scope_document_ids=scope,
    )


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: UUID, body: SessionRename, session_repo: SessionRepoDep
) -> SessionDetailOut:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    await session_repo.rename_session(session_id, title=body.title)
    return await get_session(session_id, session_repo)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: UUID, session_repo: SessionRepoDep) -> None:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    await session_repo.delete_session(session_id)


@router.post("/sessions/{session_id}/archive", status_code=204)
async def archive_session(session_id: UUID, session_repo: SessionRepoDep) -> None:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    await session_repo.archive_session(session_id)


@router.put("/sessions/{session_id}/scope", status_code=204)
async def set_session_scope(
    session_id: UUID, body: SessionScopeIn, session_repo: SessionRepoDep
) -> None:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    await session_repo.set_session_scope(session_id, body.document_ids)


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: UUID, session_repo: SessionRepoDep
) -> list[MessageOut]:
    if await session_repo.load_session(session_id) is None:
        raise NotFoundError("session", str(session_id))
    messages = await session_repo.list_messages(session_id)
    citations_by_id = await session_repo.list_citations([m.id for m in messages])
    return [MessageOut.from_row(m, citations_by_id.get(m.id, [])) for m in messages]
