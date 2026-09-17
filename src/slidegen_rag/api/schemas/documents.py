from __future__ import annotations

import dataclasses
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from slidegen_rag.memory.dto import DocumentIndexRow, DocumentRow


class DocumentRename(BaseModel):
    title: str


class DocumentOut(BaseModel):
    id: UUID
    project_id: UUID
    ref_no: int
    doc_slug: str
    title: str
    source_sha1: str
    source_path: str | None = None
    source_bytes: int | None = None
    status: str
    active_index_id: UUID | None = None
    doc_summary: str | None = None
    outline: list[str] | None = None
    total_nodes: int | None = None
    num_layers: int | None = None
    total_tokens: int | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def from_row(cls, row: DocumentRow) -> DocumentOut:
        return cls(**dataclasses.asdict(row))


class DocumentIndexOut(BaseModel):
    id: UUID
    document_id: UUID
    attempt_no: int
    status: str
    source_sha1: str
    artifact_path: str | None = None
    artifact_sha1: str | None = None
    total_nodes: int | None = None
    num_layers: int | None = None
    total_tokens: int | None = None
    build_stats: dict | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime

    @classmethod
    def from_row(cls, row: DocumentIndexRow) -> DocumentIndexOut:
        return cls(**dataclasses.asdict(row))


class DocumentDetailOut(BaseModel):
    document: DocumentOut
    active_index: DocumentIndexOut | None = None


class StartIndexOut(BaseModel):
    job_id: str
    index_id: str
