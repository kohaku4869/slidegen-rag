"""
catalog.py — Giao diện `ProjectRepository`/`DocumentRepository` (xem
plan6.md B2) cộng implementation thật `PostgresProjectRepository` /
`PostgresDocumentRepository`. Soi gương `repository.py`: Protocol trước,
implementation Postgres sau; fake in-memory nằm ở `catalog_store.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from slidegen_rag.memory.dto import (
    DocumentIndexRow,
    DocumentRow,
    ProjectRow,
    ProjectStats,
)
from slidegen_rag.memory.models import Document, DocumentIndex, Project, Session


@runtime_checkable
class ProjectRepository(Protocol):
    async def create_project(
        self, *, owner_id: UUID, name: str, language: str = "vi"
    ) -> ProjectRow: ...

    async def list_projects(
        self, *, owner_id: UUID, include_deleted: bool = False
    ) -> list[ProjectRow]: ...

    async def get_project(self, project_id: UUID) -> ProjectRow | None: ...

    async def rename_project(self, project_id: UUID, *, name: str) -> None: ...

    async def soft_delete_project(self, project_id: UUID) -> None: ...

    async def project_stats(self, project_id: UUID) -> ProjectStats: ...


@runtime_checkable
class DocumentRepository(Protocol):
    async def create_document(
        self,
        *,
        project_id: UUID,
        doc_slug: str,
        title: str,
        source_sha1: str,
        source_path: str | None = None,
        source_bytes: int | None = None,
    ) -> DocumentRow: ...

    async def find_by_sha1(
        self, *, project_id: UUID, source_sha1: str
    ) -> DocumentRow | None: ...

    async def list_documents(
        self, *, project_id: UUID, include_deleted: bool = False
    ) -> list[DocumentRow]: ...

    async def rename_document(self, document_id: UUID, *, title: str) -> None: ...

    async def soft_delete_document(self, document_id: UUID) -> None: ...

    async def start_index(
        self, *, document_id: UUID, source_sha1: str
    ) -> DocumentIndexRow: ...

    async def update_index_status(
        self, index_id: UUID, *, status: str, error_message: str | None = None
    ) -> None: ...

    async def fail_index(self, index_id: UUID, *, error_message: str) -> None: ...

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
    ) -> None: ...

    async def get_active_index(self, document_id: UUID) -> DocumentIndexRow | None: ...

    async def list_indexes(self, document_id: UUID) -> list[DocumentIndexRow]: ...

    async def get_document(self, document_id: UUID) -> DocumentRow | None: ...


def _project_row(r: Project) -> ProjectRow:
    return ProjectRow(
        id=r.id,
        owner_id=r.owner_id,
        name=r.name,
        language=r.language,
        created_at=r.created_at,
        updated_at=r.updated_at,
        deleted_at=r.deleted_at,
    )


def _document_row(r: Document) -> DocumentRow:
    return DocumentRow(
        id=r.id,
        project_id=r.project_id,
        ref_no=r.ref_no,
        doc_slug=r.doc_slug,
        title=r.title,
        source_sha1=r.source_sha1.rstrip(),
        source_path=r.source_path,
        source_bytes=r.source_bytes,
        status=r.status.value if hasattr(r.status, "value") else r.status,
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


def _index_row(r: DocumentIndex) -> DocumentIndexRow:
    return DocumentIndexRow(
        id=r.id,
        document_id=r.document_id,
        attempt_no=r.attempt_no,
        status=r.status.value if hasattr(r.status, "value") else r.status,
        source_sha1=r.source_sha1.rstrip(),
        artifact_path=r.artifact_path,
        artifact_sha1=(r.artifact_sha1.rstrip() if r.artifact_sha1 else None),
        total_nodes=r.total_nodes,
        num_layers=r.num_layers,
        total_tokens=r.total_tokens,
        build_stats=r.build_stats,
        error_message=r.error_message,
        started_at=r.started_at,
        finished_at=r.finished_at,
        created_at=r.created_at,
    )


class PostgresProjectRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    async def create_project(
        self, *, owner_id: UUID, name: str, language: str = "vi"
    ) -> ProjectRow:
        async with self._sessionmaker() as db:
            row = Project(owner_id=owner_id, name=name, language=language)
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return _project_row(row)

    async def list_projects(
        self, *, owner_id: UUID, include_deleted: bool = False
    ) -> list[ProjectRow]:
        async with self._sessionmaker() as db:
            stmt = select(Project).where(Project.owner_id == owner_id)
            if not include_deleted:
                stmt = stmt.where(Project.deleted_at.is_(None))
            stmt = stmt.order_by(Project.updated_at.desc())
            rows = (await db.execute(stmt)).scalars().all()
            return [_project_row(r) for r in rows]

    async def get_project(self, project_id: UUID) -> ProjectRow | None:
        async with self._sessionmaker() as db:
            row = await db.get(Project, project_id)
            return _project_row(row) if row is not None else None

    async def rename_project(self, project_id: UUID, *, name: str) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(Project)
                .where(Project.id == project_id)
                .values(name=name, updated_at=func.now())
            )
            await db.commit()

    async def soft_delete_project(self, project_id: UUID) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(Project)
                .where(Project.id == project_id)
                .values(deleted_at=func.now(), updated_at=func.now())
            )
            await db.commit()

    async def project_stats(self, project_id: UUID) -> ProjectStats:
        async with self._sessionmaker() as db:
            document_count = (
                await db.execute(
                    select(func.count(Document.id)).where(
                        Document.project_id == project_id,
                        Document.deleted_at.is_(None),
                    )
                )
            ).scalar_one()
            session_count = (
                await db.execute(
                    select(func.count(Session.id)).where(
                        Session.project_id == project_id
                    )
                )
            ).scalar_one()
            last_active_at = (
                await db.execute(
                    select(func.max(Session.last_active_at)).where(
                        Session.project_id == project_id
                    )
                )
            ).scalar_one()
            return ProjectStats(
                document_count=document_count,
                session_count=session_count,
                last_active_at=last_active_at,
            )


class PostgresDocumentRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

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
        async with self._sessionmaker() as db, db.begin():
            # Advisory lock để `ref_no` tất định không đụng nhau khi hai
            # upload cùng project chạy song song (xem plan6.md B1).
            await db.execute(
                select(func.pg_advisory_xact_lock(func.hashtext(str(project_id))))
            )
            next_ref = (
                await db.execute(
                    select(func.coalesce(func.max(Document.ref_no), 0) + 1).where(
                        Document.project_id == project_id
                    )
                )
            ).scalar_one()
            row = Document(
                project_id=project_id,
                ref_no=next_ref,
                doc_slug=doc_slug,
                title=title,
                source_sha1=source_sha1,
                source_path=source_path,
                source_bytes=source_bytes,
            )
            db.add(row)
            await db.flush()
            await db.refresh(row)
            return _document_row(row)

    async def find_by_sha1(
        self, *, project_id: UUID, source_sha1: str
    ) -> DocumentRow | None:
        async with self._sessionmaker() as db:
            row = (
                await db.execute(
                    select(Document).where(
                        Document.project_id == project_id,
                        Document.source_sha1 == source_sha1,
                        Document.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            return _document_row(row) if row is not None else None

    async def list_documents(
        self, *, project_id: UUID, include_deleted: bool = False
    ) -> list[DocumentRow]:
        async with self._sessionmaker() as db:
            stmt = select(Document).where(Document.project_id == project_id)
            if not include_deleted:
                stmt = stmt.where(Document.deleted_at.is_(None))
            stmt = stmt.order_by(Document.ref_no)
            rows = (await db.execute(stmt)).scalars().all()
            return [_document_row(r) for r in rows]

    async def get_document(self, document_id: UUID) -> DocumentRow | None:
        async with self._sessionmaker() as db:
            row = await db.get(Document, document_id)
            return _document_row(row) if row is not None else None

    async def rename_document(self, document_id: UUID, *, title: str) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(title=title, updated_at=func.now())
            )
            await db.commit()

    async def soft_delete_document(self, document_id: UUID) -> None:
        async with self._sessionmaker() as db:
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(deleted_at=func.now(), updated_at=func.now())
            )
            await db.commit()

    async def start_index(
        self, *, document_id: UUID, source_sha1: str
    ) -> DocumentIndexRow:
        async with self._sessionmaker() as db, db.begin():
            next_attempt = (
                await db.execute(
                    select(
                        func.coalesce(func.max(DocumentIndex.attempt_no), 0) + 1
                    ).where(DocumentIndex.document_id == document_id)
                )
            ).scalar_one()
            row = DocumentIndex(
                document_id=document_id,
                attempt_no=next_attempt,
                source_sha1=source_sha1,
                status="parsing",
                started_at=func.now(),
            )
            db.add(row)
            await db.flush()
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status="parsing", updated_at=func.now())
            )
            await db.refresh(row)
            return _index_row(row)

    async def update_index_status(
        self, index_id: UUID, *, status: str, error_message: str | None = None
    ) -> None:
        async with self._sessionmaker() as db:
            values: dict = {"status": status}
            if error_message is not None:
                values["error_message"] = error_message
            await db.execute(
                update(DocumentIndex).where(DocumentIndex.id == index_id).values(**values)
            )
            index_row = await db.get(DocumentIndex, index_id)
            if index_row is not None:
                await db.execute(
                    update(Document)
                    .where(Document.id == index_row.document_id)
                    .values(status=status, updated_at=func.now())
                )
            await db.commit()

    async def fail_index(self, index_id: UUID, *, error_message: str) -> None:
        async with self._sessionmaker() as db, db.begin():
            await db.execute(
                update(DocumentIndex)
                .where(DocumentIndex.id == index_id)
                .values(
                    status="failed",
                    error_message=error_message,
                    finished_at=func.now(),
                )
            )
            index_row = await db.get(DocumentIndex, index_id)
            document_row = await db.get(Document, index_row.document_id)
            # Có bản cũ ready thì tài liệu vẫn dùng được — trả
            # documents.status về 'ready' (start_index đã tạm ghi đè
            # thành 'parsing' lúc bắt đầu build lại); chỉ đổi 'failed'
            # khi chưa từng có active_index_id.
            document_row.status = (
                "failed" if document_row.active_index_id is None else "ready"
            )
            document_row.updated_at = func.now()

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
        async with self._sessionmaker() as db, db.begin():
            index_row = await db.get(DocumentIndex, index_id)
            index_row.status = "ready"
            index_row.artifact_path = artifact_path
            index_row.artifact_sha1 = artifact_sha1
            index_row.build_stats = build_stats
            index_row.total_nodes = total_nodes
            index_row.num_layers = num_layers
            index_row.total_tokens = total_tokens
            index_row.finished_at = func.now()

            document_row = await db.get(Document, index_row.document_id)
            previous_active_id = document_row.active_index_id
            document_row.active_index_id = index_row.id
            document_row.status = "ready"
            document_row.doc_summary = doc_summary
            document_row.outline = outline
            document_row.total_nodes = total_nodes
            document_row.num_layers = num_layers
            document_row.total_tokens = total_tokens
            document_row.updated_at = func.now()

            if previous_active_id is not None and previous_active_id != index_row.id:
                await db.execute(
                    update(DocumentIndex)
                    .where(
                        DocumentIndex.id == previous_active_id,
                        DocumentIndex.status == "ready",
                    )
                    .values(status="superseded")
                )

    async def get_active_index(self, document_id: UUID) -> DocumentIndexRow | None:
        async with self._sessionmaker() as db:
            document_row = await db.get(Document, document_id)
            if document_row is None or document_row.active_index_id is None:
                return None
            index_row = await db.get(DocumentIndex, document_row.active_index_id)
            return _index_row(index_row) if index_row is not None else None

    async def list_indexes(self, document_id: UUID) -> list[DocumentIndexRow]:
        async with self._sessionmaker() as db:
            rows = (
                await db.execute(
                    select(DocumentIndex)
                    .where(DocumentIndex.document_id == document_id)
                    .order_by(DocumentIndex.attempt_no.desc())
                )
            ).scalars().all()
            return [_index_row(r) for r in rows]
