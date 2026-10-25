from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from slidegen_rag.memory.dto import CitedNodeRef, MessageRow, SessionListRow


class SessionCreate(BaseModel):
    title: str | None = None
    document_id: UUID | None = None


class SessionRename(BaseModel):
    title: str


class SessionScopeIn(BaseModel):
    document_ids: list[UUID]


class SessionListOut(BaseModel):
    id: UUID
    title: str | None = None
    bound_doc_id: str | None = None
    status: str
    created_at: datetime
    last_active_at: datetime

    @classmethod
    def from_row(cls, row: SessionListRow) -> SessionListOut:
        return cls(
            id=row.id,
            title=row.title,
            bound_doc_id=row.bound_doc_id,
            status=row.status,
            created_at=row.created_at,
            last_active_at=row.last_active_at,
        )


class SessionDetailOut(BaseModel):
    id: UUID
    title: str | None = None
    bound_doc_id: str | None = None
    max_turns: int
    history_summary: str | None = None
    scope_document_ids: list[UUID]


class CitedNodeRefOut(BaseModel):
    doc_id: str
    source_sha1: str
    node_index: int
    node_id: int | None = None
    document_id: UUID | None = None
    priority: int | None = None

    @classmethod
    def from_ref(cls, ref: CitedNodeRef) -> CitedNodeRefOut:
        return cls(
            doc_id=ref.doc_id,
            source_sha1=ref.source_sha1,
            node_index=ref.node_index,
            node_id=ref.node_id,
            document_id=ref.document_id,
            priority=ref.priority,
        )


class MessageOut(BaseModel):
    id: UUID
    seq: int
    role: str
    content: str
    created_at: datetime
    meta: dict
    citations: list[CitedNodeRefOut] = []

    @classmethod
    def from_row(cls, row: MessageRow, citations: list[CitedNodeRef]) -> MessageOut:
        return cls(
            id=row.id,
            seq=row.seq,
            role=row.role,
            content=row.content,
            created_at=row.created_at,
            meta=row.meta,
            citations=[CitedNodeRefOut.from_ref(c) for c in citations],
        )
