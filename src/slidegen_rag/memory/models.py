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
    ARRAY,
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


class IndexStatus(str, enum.Enum):
    queued = "queued"
    parsing = "parsing"
    summarizing = "summarizing"
    embedding = "embedding"
    ready = "ready"
    failed = "failed"
    superseded = "superseded"


index_status_enum = SAEnum(IndexStatus, name="index_status", native_enum=True)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False, server_default="vi")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "projects_owner_name_uniq",
            owner_id,
            name,
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    ref_no: Mapped[int] = mapped_column(Integer, nullable=False)
    doc_slug: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha1: Mapped[str] = mapped_column(CHAR(40), nullable=False)
    source_path: Mapped[str | None] = mapped_column(Text)
    source_bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[IndexStatus] = mapped_column(
        index_status_enum, nullable=False, server_default="queued"
    )
    active_index_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_indexes.id", ondelete="SET NULL", use_alter=True),
    )
    doc_summary: Mapped[str | None] = mapped_column(Text)
    outline: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    total_nodes: Mapped[int | None] = mapped_column(Integer)
    num_layers: Mapped[int | None] = mapped_column(SmallInteger)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="documents_pid_id_uniq"),
        UniqueConstraint("project_id", "ref_no", name="documents_pid_ref_uniq"),
        Index(
            "documents_project_sha1_uniq",
            project_id,
            source_sha1,
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )


class DocumentIndex(Base):
    __tablename__ = "document_indexes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    status: Mapped[IndexStatus] = mapped_column(
        index_status_enum, nullable=False, server_default="queued"
    )
    source_sha1: Mapped[str] = mapped_column(CHAR(40), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    artifact_sha1: Mapped[str | None] = mapped_column(CHAR(40))
    total_nodes: Mapped[int | None] = mapped_column(Integer)
    num_layers: Mapped[int | None] = mapped_column(SmallInteger)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    build_stats: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("document_id", "attempt_no", name="document_indexes_attempt_uniq"),
    )


class SessionDocumentScope(Base):
    __tablename__ = "session_document_scope"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuidv7()")
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE", use_alter=True)
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL", use_alter=True)
    )
    title: Mapped[str | None] = mapped_column(Text)
    # TẠM: doc_id/provenance .pkl mà phiên CLI này gắn vào — DROP ở migration
    # 0003 sau khi scripts/adopt_pkl_indexes.py đã tạo documents cho mọi .pkl.
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
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL", use_alter=True)
    )
    priority: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("message_citations_doc", doc_id, node_index),)
