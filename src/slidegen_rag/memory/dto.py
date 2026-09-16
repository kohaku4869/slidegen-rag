"""
dto.py — Kiểu dữ liệu bất biến (frozen dataclass) truyền giữa
`ConversationManager`, `SessionRepository` và CLI. Không phụ thuộc
SQLAlchemy hay asyncpg — để `InMemorySessionRepository` dùng được mà không
kéo theo DB driver nào.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """Danh tính tài liệu đang nạp ở phiên CLI hiện tại — dùng để lọc citation
    khi resume (xem B1 điểm 1 trong plan4.md): tree build lại thì sha1 đổi,
    citation cũ không còn khớp node hiện tại."""

    doc_id: str
    source_sha1: str


@dataclass(frozen=True, slots=True)
class CitedNodeRef:
    """Một node được trích dẫn trong câu trả lời. Định danh bằng bộ ba
    (doc_id, source_sha1, node_index) — KHÔNG phải `node_id` int trần, vì
    `Node.index` chỉ duy nhất trong một cây (xem plan4.md B1.1)."""

    doc_id: str
    source_sha1: str
    node_index: int
    node_id: int | None = None
    document_id: UUID | None = None
    priority: int | None = None


@dataclass(frozen=True, slots=True)
class TurnEntry:
    seq: int
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str


@dataclass(frozen=True, slots=True)
class SessionState:
    id: UUID
    bound_doc_id: str | None
    max_turns: int
    history_summary: str | None
    compacted_through_seq: int
    next_seq: int
    entries: list[TurnEntry] = field(default_factory=list)
    last_cited: list[CitedNodeRef] = field(default_factory=list)
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SessionListRow:
    id: UUID
    title: str | None
    bound_doc_id: str | None
    status: str
    created_at: datetime
    last_active_at: datetime


@dataclass(frozen=True, slots=True)
class MessageRow:
    id: UUID
    seq: int
    role: str
    content: str
    created_at: datetime
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectRow:
    id: UUID
    owner_id: UUID
    name: str
    language: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProjectStats:
    document_count: int
    session_count: int
    last_active_at: datetime | None


@dataclass(frozen=True, slots=True)
class DocumentRow:
    id: UUID
    project_id: UUID
    ref_no: int
    doc_slug: str
    title: str
    source_sha1: str
    source_path: str | None
    source_bytes: int | None
    status: str
    active_index_id: UUID | None
    doc_summary: str | None
    outline: list[str] | None
    total_nodes: int | None
    num_layers: int | None
    total_tokens: int | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentIndexRow:
    id: UUID
    document_id: UUID
    attempt_no: int
    status: str
    source_sha1: str
    artifact_path: str | None
    artifact_sha1: str | None
    total_nodes: int | None
    num_layers: int | None
    total_tokens: int | None
    build_stats: dict | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
