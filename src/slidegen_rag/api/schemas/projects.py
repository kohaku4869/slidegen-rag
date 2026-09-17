from __future__ import annotations

import dataclasses
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from slidegen_rag.memory.dto import ProjectRow, ProjectStats


class ProjectCreate(BaseModel):
    name: str
    language: str = "vi"


class ProjectRename(BaseModel):
    name: str


class ProjectOut(BaseModel):
    id: UUID
    owner_id: UUID
    name: str
    language: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def from_row(cls, row: ProjectRow) -> ProjectOut:
        return cls(**dataclasses.asdict(row))


class ProjectStatsOut(BaseModel):
    document_count: int
    session_count: int
    last_active_at: datetime | None = None

    @classmethod
    def from_row(cls, row: ProjectStats) -> ProjectStatsOut:
        return cls(**dataclasses.asdict(row))
