"""
repository.py — Giao diện `SessionRepository` mà `ConversationManager` (và
sau này API server) code against, cộng implementation thật
`PostgresSessionRepository`. Implementation fake/fallback nằm ở
`memory_store.InMemorySessionRepository` (dùng khi không có DATABASE_URL).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from slidegen_rag.memory.dto import (
    CitedNodeRef,
    SessionListRow,
    SessionState,
    TurnEntry,
)
from slidegen_rag.memory.models import (
    Message,
    MessageCitation,
    MessageRole,
    Session,
    SessionMemory,
)


@runtime_checkable
class SessionRepository(Protocol):
    async def create_session(
        self,
        *,
        bound_doc_id: str | None = None,
        index_path: str | None = None,
        title: str | None = None,
        max_turns: int = 5,
        project_id: UUID | None = None,
    ) -> SessionState: ...

    async def load_session(self, session_id: UUID) -> SessionState | None: ...

    async def latest_session(
        self, *, bound_doc_id: str | None = None
    ) -> SessionState | None: ...

    async def list_sessions(
        self, *, bound_doc_id: str | None = None, limit: int = 20
    ) -> list[SessionListRow]: ...

    async def append_turn(
        self,
        session_id: UUID,
        *,
        raw_query: str,
        standalone_query: str,
        answer: str,
        cited: Sequence[CitedNodeRef],
    ) -> tuple[int, int]: ...

    async def save_compaction(
        self, session_id: UUID, *, summary: str, compacted_through_seq: int
    ) -> None: ...

    async def archive_session(self, session_id: UUID) -> None: ...


def _eq_or_is_null(column, value):
    """`column == None` sinh ra SQL `= NULL` (luôn false) — cần `IS NULL`
    tường minh khi lọc `bound_doc_id` rỗng (phiên chưa gắn tài liệu nào)."""
    return column.is_(None) if value is None else column == value


class PostgresSessionRepository:
    """Implementation thật của `SessionRepository`, dùng SQLAlchemy 2.0 async
    + asyncpg. Cần `models.py` đã migrate (`alembic upgrade head`)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    async def create_session(
        self,
        *,
        bound_doc_id: str | None = None,
        index_path: str | None = None,
        title: str | None = None,
        max_turns: int = 5,
        project_id: UUID | None = None,
    ) -> SessionState:
        async with self._sessionmaker() as db:
            session_row = Session(
                bound_doc_id=bound_doc_id,
                index_path=index_path,
                title=title,
                project_id=project_id,
            )
            db.add(session_row)
            await db.flush()
            db.add(SessionMemory(session_id=session_row.id, max_turns=max_turns))
            await db.commit()
            return SessionState(
                id=session_row.id,
                bound_doc_id=bound_doc_id,
                max_turns=max_turns,
                history_summary=None,
                compacted_through_seq=0,
                next_seq=1,
                entries=[],
                last_cited=[],
            )

    async def load_session(self, session_id: UUID) -> SessionState | None:
        async with self._sessionmaker() as db:
            return await self._load_session(db, session_id)

    async def _load_session(
        self, db: AsyncSession, session_id: UUID
    ) -> SessionState | None:
        row = (
            await db.execute(
                select(Session, SessionMemory)
                .join(SessionMemory, SessionMemory.session_id == Session.id)
                .where(Session.id == session_id)
            )
        ).first()
        if row is None:
            return None
        session_row, memory_row = row
        entries = await self._load_entries(
            db, session_id, memory_row.compacted_through_seq
        )
        last_cited = await self._load_last_cited(db, session_id)
        return SessionState(
            id=session_row.id,
            bound_doc_id=session_row.bound_doc_id,
            max_turns=memory_row.max_turns,
            history_summary=memory_row.history_summary,
            compacted_through_seq=memory_row.compacted_through_seq,
            next_seq=session_row.next_seq,
            entries=entries,
            last_cited=last_cited,
        )

    async def _load_entries(
        self, db: AsyncSession, session_id: UUID, compacted_through_seq: int
    ) -> list[TurnEntry]:
        rows = (
            await db.execute(
                select(Message.seq, Message.role, Message.content)
                .where(
                    Message.session_id == session_id,
                    Message.seq > compacted_through_seq,
                )
                .order_by(Message.seq)
            )
        ).all()
        return [
            TurnEntry(
                seq=r.seq,
                role=r.role.value if isinstance(r.role, MessageRole) else r.role,
                content=r.content,
            )
            for r in rows
        ]

    async def _load_last_cited(
        self, db: AsyncSession, session_id: UUID
    ) -> list[CitedNodeRef]:
        last_message_id = (
            await db.execute(
                select(Message.id)
                .where(
                    Message.session_id == session_id,
                    Message.role == MessageRole.assistant,
                )
                .order_by(Message.seq.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if last_message_id is None:
            return []
        rows = (
            (
                await db.execute(
                    select(MessageCitation)
                    .where(MessageCitation.message_id == last_message_id)
                    .order_by(MessageCitation.rank)
                )
            )
            .scalars()
            .all()
        )
        return [
            CitedNodeRef(
                doc_id=r.doc_id,
                # CHAR(40) Postgres trả về đệm space bên phải — vô hại với
                # sha1 hex thật (đúng 40 ký tự) nhưng rstrip để "" (tree v1
                # chưa có sha1) không bị lệch khỏi DocumentRef.source_sha1="".
                source_sha1=r.source_sha1.rstrip(),
                node_index=r.node_index,
                node_id=r.node_id,
                document_id=r.document_id,
                priority=r.priority,
            )
            for r in rows
        ]

    async def latest_session(
        self, *, bound_doc_id: str | None = None
    ) -> SessionState | None:
        async with self._sessionmaker() as db:
            session_id = (
                await db.execute(
                    select(Session.id)
                    .where(
                        Session.status == "active",
                        _eq_or_is_null(Session.bound_doc_id, bound_doc_id),
                    )
                    .order_by(Session.last_active_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if session_id is None:
                return None
            return await self._load_session(db, session_id)

    async def list_sessions(
        self, *, bound_doc_id: str | None = None, limit: int = 20
    ) -> list[SessionListRow]:
        async with self._sessionmaker() as db:
            stmt = select(Session).order_by(Session.last_active_at.desc()).limit(limit)
            if bound_doc_id is not None:
                stmt = stmt.where(Session.bound_doc_id == bound_doc_id)
            rows = (await db.execute(stmt)).scalars().all()
        return [
            SessionListRow(
                id=r.id,
                title=r.title,
                bound_doc_id=r.bound_doc_id,
                status=r.status,
                created_at=r.created_at,
                last_active_at=r.last_active_at,
            )
            for r in rows
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
        async with self._sessionmaker() as db:
            async with db.begin():
                next_seq = (
                    await db.execute(
                        select(Session.next_seq)
                        .where(Session.id == session_id)
                        .with_for_update()
                    )
                ).scalar_one()
                user_seq, assistant_seq = next_seq, next_seq + 1
                await db.execute(
                    update(Session)
                    .where(Session.id == session_id)
                    .values(next_seq=next_seq + 2, last_active_at=func.now())
                )
                user_msg = Message(
                    session_id=session_id,
                    seq=user_seq,
                    role=MessageRole.user,
                    content=raw_query,
                    raw_query=raw_query,
                    standalone_query=standalone_query,
                )
                assistant_msg = Message(
                    session_id=session_id,
                    seq=assistant_seq,
                    role=MessageRole.assistant,
                    content=answer,
                )
                db.add_all([user_msg, assistant_msg])
                await db.flush()
                for rank, ref in enumerate(cited, start=1):
                    db.add(
                        MessageCitation(
                            message_id=assistant_msg.id,
                            rank=rank,
                            doc_id=ref.doc_id,
                            source_sha1=ref.source_sha1,
                            node_index=ref.node_index,
                            node_id=ref.node_id,
                            document_id=ref.document_id,
                            priority=ref.priority,
                        )
                    )
            return user_seq, assistant_seq

    async def save_compaction(
        self, session_id: UUID, *, summary: str, compacted_through_seq: int
    ) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(SessionMemory)
                .where(
                    SessionMemory.session_id == session_id,
                    SessionMemory.compacted_through_seq < compacted_through_seq,
                )
                .values(
                    history_summary=summary,
                    compacted_through_seq=compacted_through_seq,
                    updated_at=func.now(),
                )
            )
            await db.commit()

    async def archive_session(self, session_id: UUID) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(Session)
                .where(Session.id == session_id)
                .values(status="archived")
            )
            await db.commit()
