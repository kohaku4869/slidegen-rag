"""
models.py — SQLAlchemy 2.0 declarative models cho migration `0001_session_memory`
(xem plan4.md, phần B1). Tên bảng/cột khớp Y HỆT schema đích ở phần A2 của
plan để lần Phần A lên chỉ cần `ALTER ... ADD CONSTRAINT`, không đổi tên hay
dựng lại bảng.

`sessions.id`/`messages.id` dùng `uuidv7()` (native trong Postgres 18+, máy
dev đang chạy PG18) thay vì sinh UUIDv7 phía Python (`uuid6`) — giữ tính cục
bộ B-tree mà UUIDv4 phá vỡ mà không cần thêm dependency.
"""

from __future__ import annotations

import datetime
import enum
import uuid

from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


message_role_enum = SAEnum(MessageRole, name="message_role", native_enum=True)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    # FK -> projects(id)/users(id) thêm ở migration sau (Phần A chưa tạo bảng)
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    title: Mapped[str | None] = mapped_column(Text)
    # TẠM: doc_id/provenance .pkl mà phiên CLI này gắn vào — DROP khi có
    # bảng documents/projects (session thuộc project, không thuộc 1 tài liệu)
    bound_doc_id: Mapped[str | None] = mapped_column(Text)
    index_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    next_seq: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_active_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active','archived')", name="sessions_status_check"
        ),
        Index(
            "sessions_recent",
            bound_doc_id,
            last_active_at.desc(),
            postgresql_where=text("status = 'active'"),
        ),
    )


class SessionMemory(Base):
    __tablename__ = "session_memory"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    max_turns: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="5"
    )
    history_summary: Mapped[str | None] = mapped_column(Text)
    compacted_through_seq: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[MessageRole] = mapped_column(message_role_enum, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_query: Mapped[str | None] = mapped_column(Text)
    standalone_query: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="messages_seq_uniq"),
        Index("messages_session_seq", session_id, seq.desc()),
    )


class MessageCitation(Base):
    __tablename__ = "message_citations"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    # Chưa có bảng nodes -> định danh node bằng bộ ba text, không phải int
    # trần (xem plan4.md B1.1). node_id/document_id backfill ở migration sau.
    doc_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha1: Mapped[str] = mapped_column(
        CHAR(40), nullable=False, server_default=""
    )
    node_index: Mapped[int] = mapped_column(Integer, nullable=False)
    node_id: Mapped[int | None] = mapped_column(BigInteger)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    priority: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("message_citations_doc", doc_id, node_index),)
