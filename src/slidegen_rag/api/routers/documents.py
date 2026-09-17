from __future__ import annotations

import hashlib
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile

from slidegen_rag.api.deps import DocumentRepoDep, JobRegistryDep, ProjectRepoDep
from slidegen_rag.api.errors import ConflictError, NotFoundError
from slidegen_rag.api.schemas.documents import (
    DocumentDetailOut,
    DocumentIndexOut,
    DocumentOut,
    DocumentRename,
    StartIndexOut,
)
from slidegen_rag.api.services import ingest_service

router = APIRouter(prefix="/api/v1", tags=["documents"])


def _slugify(title: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
    return slug or "document"


@router.post("/projects/{project_id}/documents", status_code=201)
async def upload_document(
    project_id: UUID,
    project_repo: ProjectRepoDep,
    document_repo: DocumentRepoDep,
    title: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> DocumentOut:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    content = await file.read()
    markdown_text = content.decode("utf-8", errors="ignore")
    source_sha1 = hashlib.sha1(content).hexdigest()

    existing = await document_repo.find_by_sha1(
        project_id=project_id, source_sha1=source_sha1
    )
    if existing is not None:
        return DocumentOut.from_row(existing)

    row = await document_repo.create_document(
        project_id=project_id,
        doc_slug=_slugify(title),
        title=title,
        source_sha1=source_sha1,
        source_path=file.filename,
        source_bytes=len(content),
    )
    # Lưu nội dung ra đĩa ngay để `POST .../index` dùng lại mà không cần
    # client gửi lại markdown lần nữa.
    src_path = ingest_service.source_md_path(project_id, row.id)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(markdown_text, encoding="utf-8")
    return DocumentOut.from_row(row)


@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID, document_repo: DocumentRepoDep
) -> DocumentDetailOut:
    row = await document_repo.get_document(document_id)
    if row is None:
        raise NotFoundError("document", str(document_id))
    active_index = await document_repo.get_active_index(document_id)
    return DocumentDetailOut(
        document=DocumentOut.from_row(row),
        active_index=DocumentIndexOut.from_row(active_index) if active_index else None,
    )


@router.patch("/documents/{document_id}")
async def rename_document(
    document_id: UUID, body: DocumentRename, document_repo: DocumentRepoDep
) -> DocumentOut:
    if await document_repo.get_document(document_id) is None:
        raise NotFoundError("document", str(document_id))
    await document_repo.rename_document(document_id, title=body.title)
    row = await document_repo.get_document(document_id)
    return DocumentOut.from_row(row)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: UUID, document_repo: DocumentRepoDep) -> None:
    if await document_repo.get_document(document_id) is None:
        raise NotFoundError("document", str(document_id))
    await document_repo.soft_delete_document(document_id)


@router.post("/documents/{document_id}/index", status_code=202)
async def start_index(
    document_id: UUID,
    document_repo: DocumentRepoDep,
    job_registry: JobRegistryDep,
) -> StartIndexOut:
    document = await document_repo.get_document(document_id)
    if document is None:
        raise NotFoundError("document", str(document_id))
    if document.status == "parsing":
        raise ConflictError(
            f"Document {document_id} đang được index, chưa xong lần trước"
        )
    if document.source_path is None:
        raise ConflictError(
            f"Document {document_id} chưa có nội dung markdown để index"
        )

    src_path = ingest_service.source_md_path(document.project_id, document.id)
    if not src_path.exists():
        raise ConflictError(
            f"Không tìm thấy nội dung markdown đã lưu cho document {document_id}"
        )
    markdown_text = src_path.read_text(encoding="utf-8")

    job = await ingest_service.start_ingest_job(
        document_repo,
        project_id=document.project_id,
        document=document,
        markdown_text=markdown_text,
        job_registry=job_registry,
    )
    return StartIndexOut(job_id=job.job_id, index_id=job.index_id)


@router.get("/documents/{document_id}/indexes")
async def list_indexes(
    document_id: UUID, document_repo: DocumentRepoDep
) -> list[DocumentIndexOut]:
    if await document_repo.get_document(document_id) is None:
        raise NotFoundError("document", str(document_id))
    rows = await document_repo.list_indexes(document_id)
    return [DocumentIndexOut.from_row(r) for r in rows]
