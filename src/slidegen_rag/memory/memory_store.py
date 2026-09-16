"""
memory_store.py — `InMemorySessionRepository`: implementation thật thứ hai
của `SessionRepository` (Protocol ở repository.py), không phải mock. Dùng
làm fallback production khi thiếu `DATABASE_URL` (UX y hệt hôm nay — mất khi
đóng terminal) VÀ làm fake trong test, để test `ConversationManager` chạy
được mà không cần Postgres (xem plan4.md, mục Verification > Chiến lược
test).

Mô phỏng lại đúng 3 ngữ nghĩa mà schema DB có, cố tình KHÔNG đơn giản hoá đi
để test trên đây có giá trị: cấp `seq` tuần tự dưới lock (không dùng
`created_at` để sắp thứ tự), watermark `compacted_through_seq` thay vì cắt
list message, và `save_compaction` idempotent (không lùi watermark).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from slidegen_rag.memory.dto import (
    CitedNodeRef,
    SessionListRow,
    SessionState,
    TurnEntry,
)


@dataclass
class _StoredMessage:
    seq: int
    role: str
    content: str


@dataclass
class _StoredSession:
    id: UUID
    bound_doc_id: str | None
    index_path: str | None
    title: str | None
    max_turns: int
    project_id: UUID | None
    status: str = "active"
    next_seq: int = 1
    history_summary: str | None = None
    compacted_through_seq: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    messages: list[_StoredMessage] = field(default_factory=list)
    citations: dict[int, list[CitedNodeRef]] = field(default_factory=dict)


class InMemorySessionRepository:
    def __init__(self) -> None:
        self._sessions: dict[UUID, _StoredSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        *,
        bound_doc_id: str | None = None,
        index_path: str | None = None,
        title: str | None = None,
        max_turns: int = 5,
        project_id: UUID | None = None,
    ) -> SessionState:
        async with self._lock:
            session_id = uuid4()
            row = _StoredSession(
                id=session_id,
                bound_doc_id=bound_doc_id,
                index_path=index_path,
                title=title,
                max_turns=max_turns,
                project_id=project_id,
            )
            self._sessions[session_id] = row
            return self._to_state(row)

    async def load_session(self, session_id: UUID) -> SessionState | None:
        row = self._sessions.get(session_id)
        return self._to_state(row) if row is not None else None

    async def latest_session(
        self, *, bound_doc_id: str | None = None
    ) -> SessionState | None:
        candidates = [
            r
            for r in self._sessions.values()
            if r.status == "active" and r.bound_doc_id == bound_doc_id
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda r: r.last_active_at)
        return self._to_state(latest)

    async def list_sessions(
        self, *, bound_doc_id: str | None = None, limit: int = 20
    ) -> list[SessionListRow]:
        rows = list(self._sessions.values())
        if bound_doc_id is not None:
            rows = [r for r in rows if r.bound_doc_id == bound_doc_id]
        rows.sort(key=lambda r: r.last_active_at, reverse=True)
        return [
            SessionListRow(
                id=r.id,
                title=r.title,
                bound_doc_id=r.bound_doc_id,
                status=r.status,
                created_at=r.created_at,
                last_active_at=r.last_active_at,
            )
            for r in rows[:limit]
        ]

    async def append_turn(
        self,
        session_id: UUID,
        *,
        raw_query: str,
        standalone_query: str,
        answer: str,
        cited: Sequence[CitedNodeRef],
    ) -> tuple[int, int]:
        async with self._lock:
            row = self._sessions[session_id]
            user_seq = row.next_seq
            assistant_seq = row.next_seq + 1
            row.next_seq += 2
            row.messages.append(
                _StoredMessage(seq=user_seq, role="user", content=raw_query)
            )
            row.messages.append(
                _StoredMessage(seq=assistant_seq, role="assistant", content=answer)
            )
            row.citations[assistant_seq] = list(cited)
            row.last_active_at = datetime.now(UTC)
            return user_seq, assistant_seq

    async def save_compaction(
        self, session_id: UUID, *, summary: str, compacted_through_seq: int
    ) -> None:
        async with self._lock:
            row = self._sessions[session_id]
            if compacted_through_seq <= row.compacted_through_seq:
                return
            row.history_summary = summary
            row.compacted_through_seq = compacted_through_seq

    async def archive_session(self, session_id: UUID) -> None:
        async with self._lock:
            self._sessions[session_id].status = "archived"

    def _to_state(self, row: _StoredSession) -> SessionState:
        entries = [
            TurnEntry(seq=m.seq, role=m.role, content=m.content)
            for m in row.messages
            if m.seq > row.compacted_through_seq
        ]
        last_assistant_seq = max(
            (m.seq for m in row.messages if m.role == "assistant"), default=None
        )
        last_cited = (
            row.citations.get(last_assistant_seq, []) if last_assistant_seq else []
        )
        return SessionState(
            id=row.id,
            bound_doc_id=row.bound_doc_id,
            max_turns=row.max_turns,
            history_summary=row.history_summary,
            compacted_through_seq=row.compacted_through_seq,
            next_seq=row.next_seq,
            entries=entries,
            last_cited=list(last_cited),
        )
