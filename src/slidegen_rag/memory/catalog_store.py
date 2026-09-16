"""
catalog_store.py — `InMemoryProjectRepository`/`InMemoryDocumentRepository`:
implementation thật thứ hai của `ProjectRepository`/`DocumentRepository`
(xem `catalog.py`), dùng làm fallback khi thiếu `DATABASE_URL` và làm fake
trong test (soi gương `memory_store.py`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from slidegen_rag.memory.dto import (
    DocumentIndexRow,
    DocumentRow,
    ProjectRow,
    ProjectStats,
)


@dataclass
class _StoredProject:
    id: UUID
    owner_id: UUID
    name: str
    language: str = "vi"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None


@dataclass
class _StoredDocument:
    id: UUID
    project_id: UUID
    ref_no: int
    doc_slug: str
    title: str
    source_sha1: str
    source_path: str | None = None
    source_bytes: int | None = None
    status: str = "queued"
    active_index_id: UUID | None = None
    doc_summary: str | None = None
    outline: list[str] | None = None
    total_nodes: int | None = None
    num_layers: int | None = None
    total_tokens: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None


@dataclass
class _StoredIndex:
    id: UUID
    document_id: UUID
    attempt_no: int
    source_sha1: str
    status: str = "queued"
    artifact_path: str | None = None
    artifact_sha1: str | None = None
    total_nodes: int | None = None
    num_layers: int | None = None
    total_tokens: int | None = None
    build_stats: dict | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _project_row(r: _StoredProject) -> ProjectRow:
    return ProjectRow(
        id=r.id,
        owner_id=r.owner_id,
        name=r.name,
        language=r.language,
        created_at=r.created_at,
        updated_at=r.updated_at,
        deleted_at=r.deleted_at,
    )


def _document_row(r: _StoredDocument) -> DocumentRow:
    return DocumentRow(
        id=r.id,
        project_id=r.project_id,
        ref_no=r.ref_no,
        doc_slug=r.doc_slug,
        title=r.title,
        source_sha1=r.source_sha1,
        source_path=r.source_path,
        source_bytes=r.source_bytes,
        status=r.status,
        active_index_id=r.active_index_id,
        doc_summary=r.doc_summary,
        outline=list(r.outline) if r.outline is not None else None,
        total_nodes=r.total_nodes,
        num_layers=r.num_layers,
        total_tokens=r.total_tokens,
        created_at=r.created_at,
        updated_at=r.updated_at,
        deleted_at=r.deleted_at,
    )


def _index_row(r: _StoredIndex) -> DocumentIndexRow:
    return DocumentIndexRow(
        id=r.id,
        document_id=r.document_id,
        attempt_no=r.attempt_no,
        status=r.status,
        source_sha1=r.source_sha1,
        artifact_path=r.artifact_path,
        artifact_sha1=r.artifact_sha1,
        total_nodes=r.total_nodes,
        num_layers=r.num_layers,
        total_tokens=r.total_tokens,
        build_stats=r.build_stats,
        error_message=r.error_message,
        started_at=r.started_at,
        finished_at=r.finished_at,
        created_at=r.created_at,
    )


class InMemoryProjectRepository:
    """`document_repo`/`session_repo` là tham chiếu tuỳ chọn tới các kho
    in-memory khác trong cùng process — chỉ `project_stats` cần chúng (gộp
    số liệu tài liệu + phiên chat, giống 1 query GROUP BY ở bản Postgres)."""

    def __init__(self, *, document_repo=None, session_repo=None) -> None:
        self._projects: dict[UUID, _StoredProject] = {}
        self._lock = asyncio.Lock()
        self._document_repo = document_repo
        self._session_repo = session_repo

    async def create_project(
        self, *, owner_id: UUID, name: str, language: str = "vi"
    ) -> ProjectRow:
        async with self._lock:
            row = _StoredProject(id=uuid4(), owner_id=owner_id, name=name, language=language)
            self._projects[row.id] = row
            return _project_row(row)

    async def list_projects(
        self, *, owner_id: UUID, include_deleted: bool = False
    ) -> list[ProjectRow]:
        rows = [
            r
            for r in self._projects.values()
            if r.owner_id == owner_id and (include_deleted or r.deleted_at is None)
        ]
        rows.sort(key=lambda r: r.updated_at, reverse=True)
        return [_project_row(r) for r in rows]

    async def get_project(self, project_id: UUID) -> ProjectRow | None:
        row = self._projects.get(project_id)
        return _project_row(row) if row is not None else None

    async def rename_project(self, project_id: UUID, *, name: str) -> None:
        async with self._lock:
            row = self._projects[project_id]
            row.name = name
            row.updated_at = datetime.now(UTC)

    async def soft_delete_project(self, project_id: UUID) -> None:
        async with self._lock:
            row = self._projects[project_id]
            row.deleted_at = datetime.now(UTC)
            row.updated_at = row.deleted_at

    async def project_stats(self, project_id: UUID) -> ProjectStats:
        document_count = 0
        if self._document_repo is not None:
            docs = await self._document_repo.list_documents(project_id=project_id)
            document_count = len(docs)
        session_count = 0
        last_active_at = None
        if self._session_repo is not None:
            sessions = await self._session_repo.list_sessions(
                project_id=project_id, limit=10_000, include_archived=True
            )
            session_count = len(sessions)
            if sessions:
                last_active_at = max(s.last_active_at for s in sessions)
        return ProjectStats(
            document_count=document_count,
            session_count=session_count,
            last_active_at=last_active_at,
        )


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[UUID, _StoredDocument] = {}
        self._indexes: dict[UUID, _StoredIndex] = {}
        self._lock = asyncio.Lock()

    async def create_document(
        self,
        *,
        project_id: UUID,
        doc_slug: str,
        title: str,
        source_sha1: str,
        source_path: str | None = None,
        source_bytes: int | None = None,
    ) -> DocumentRow:
        async with self._lock:
            existing_refs = [
                r.ref_no for r in self._documents.values() if r.project_id == project_id
            ]
            row = _StoredDocument(
                id=uuid4(),
                project_id=project_id,
                ref_no=(max(existing_refs) + 1) if existing_refs else 1,
                doc_slug=doc_slug,
                title=title,
                source_sha1=source_sha1,
                source_path=source_path,
                source_bytes=source_bytes,
            )
            self._documents[row.id] = row
            return _document_row(row)

    async def find_by_sha1(
        self, *, project_id: UUID, source_sha1: str
    ) -> DocumentRow | None:
        for r in self._documents.values():
            if (
                r.project_id == project_id
                and r.source_sha1 == source_sha1
                and r.deleted_at is None
            ):
                return _document_row(r)
        return None

    async def list_documents(
        self, *, project_id: UUID, include_deleted: bool = False
    ) -> list[DocumentRow]:
        rows = [
            r
            for r in self._documents.values()
            if r.project_id == project_id and (include_deleted or r.deleted_at is None)
        ]
        rows.sort(key=lambda r: r.ref_no)
        return [_document_row(r) for r in rows]

    async def get_document(self, document_id: UUID) -> DocumentRow | None:
        row = self._documents.get(document_id)
        return _document_row(row) if row is not None else None

    async def rename_document(self, document_id: UUID, *, title: str) -> None:
        async with self._lock:
            row = self._documents[document_id]
            row.title = title
            row.updated_at = datetime.now(UTC)

    async def soft_delete_document(self, document_id: UUID) -> None:
        async with self._lock:
            row = self._documents[document_id]
            row.deleted_at = datetime.now(UTC)
            row.updated_at = row.deleted_at

    async def start_index(
        self, *, document_id: UUID, source_sha1: str
    ) -> DocumentIndexRow:
        async with self._lock:
            attempts = [
                r.attempt_no
                for r in self._indexes.values()
                if r.document_id == document_id
            ]
            row = _StoredIndex(
                id=uuid4(),
                document_id=document_id,
                attempt_no=(max(attempts) + 1) if attempts else 1,
                source_sha1=source_sha1,
                status="parsing",
                started_at=datetime.now(UTC),
            )
            self._indexes[row.id] = row
            self._documents[document_id].status = "parsing"
            self._documents[document_id].updated_at = datetime.now(UTC)
            return _index_row(row)

    async def update_index_status(
        self, index_id: UUID, *, status: str, error_message: str | None = None
    ) -> None:
        async with self._lock:
            row = self._indexes[index_id]
            row.status = status
            if error_message is not None:
                row.error_message = error_message
            doc = self._documents.get(row.document_id)
            if doc is not None:
                doc.status = status
                doc.updated_at = datetime.now(UTC)

    async def fail_index(self, index_id: UUID, *, error_message: str) -> None:
        async with self._lock:
            row = self._indexes[index_id]
            row.status = "failed"
            row.error_message = error_message
            row.finished_at = datetime.now(UTC)
            doc = self._documents[row.document_id]
            # Chưa có bản ready nào -> tài liệu thật sự hỏng. Đã có bản cũ ->
            # tài liệu vẫn dùng được, trả documents.status về 'ready' (start_index
            # đã tạm ghi đè thành 'parsing' lúc bắt đầu build lại).
            doc.status = "failed" if doc.active_index_id is None else "ready"
            doc.updated_at = datetime.now(UTC)

    async def finish_index(
        self,
        index_id: UUID,
        *,
        artifact_path: str,
        artifact_sha1: str,
        build_stats: dict,
        doc_summary: str,
        outline: list[str],
        total_nodes: int,
        num_layers: int,
        total_tokens: int,
    ) -> None:
        async with self._lock:
            row = self._indexes[index_id]
            row.status = "ready"
            row.artifact_path = artifact_path
            row.artifact_sha1 = artifact_sha1
            row.build_stats = build_stats
            row.total_nodes = total_nodes
            row.num_layers = num_layers
            row.total_tokens = total_tokens
            row.finished_at = datetime.now(UTC)

            doc = self._documents[row.document_id]
            previous_active_id = doc.active_index_id
            doc.active_index_id = row.id
            doc.status = "ready"
            doc.doc_summary = doc_summary
            doc.outline = outline
            doc.total_nodes = total_nodes
            doc.num_layers = num_layers
            doc.total_tokens = total_tokens
            doc.updated_at = datetime.now(UTC)

            if previous_active_id is not None and previous_active_id != row.id:
                previous = self._indexes.get(previous_active_id)
                if previous is not None and previous.status == "ready":
                    previous.status = "superseded"

    async def get_active_index(self, document_id: UUID) -> DocumentIndexRow | None:
        doc = self._documents.get(document_id)
        if doc is None or doc.active_index_id is None:
            return None
        row = self._indexes.get(doc.active_index_id)
        return _index_row(row) if row is not None else None

    async def list_indexes(self, document_id: UUID) -> list[DocumentIndexRow]:
        rows = [r for r in self._indexes.values() if r.document_id == document_id]
        rows.sort(key=lambda r: r.attempt_no, reverse=True)
        return [_index_row(r) for r in rows]
