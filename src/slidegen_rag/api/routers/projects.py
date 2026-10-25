from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from slidegen_rag.api.deps import DocumentRepoDep, ProjectRepoDep
from slidegen_rag.api.errors import NotFoundError
from slidegen_rag.api.schemas.documents import DocumentOut
from slidegen_rag.api.schemas.projects import (
    ProjectCreate,
    ProjectOut,
    ProjectRename,
    ProjectStatsOut,
)
from slidegen_rag.config import LOCAL_USER_ID

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", status_code=201)
async def create_project(
    body: ProjectCreate, project_repo: ProjectRepoDep
) -> ProjectOut:
    row = await project_repo.create_project(
        owner_id=LOCAL_USER_ID, name=body.name, language=body.language
    )
    return ProjectOut.from_row(row)


@router.get("")
async def list_projects(project_repo: ProjectRepoDep) -> list[ProjectOut]:
    rows = await project_repo.list_projects(owner_id=LOCAL_USER_ID)
    return [ProjectOut.from_row(r) for r in rows]


@router.get("/{project_id}")
async def get_project(project_id: UUID, project_repo: ProjectRepoDep) -> ProjectOut:
    row = await project_repo.get_project(project_id)
    if row is None:
        raise NotFoundError("project", str(project_id))
    return ProjectOut.from_row(row)


@router.patch("/{project_id}")
async def rename_project(
    project_id: UUID, body: ProjectRename, project_repo: ProjectRepoDep
) -> ProjectOut:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    await project_repo.rename_project(project_id, name=body.name)
    row = await project_repo.get_project(project_id)
    return ProjectOut.from_row(row)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: UUID, project_repo: ProjectRepoDep) -> None:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    await project_repo.soft_delete_project(project_id)


@router.get("/{project_id}/stats")
async def get_project_stats(
    project_id: UUID, project_repo: ProjectRepoDep
) -> ProjectStatsOut:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    stats = await project_repo.project_stats(project_id)
    return ProjectStatsOut.from_row(stats)


@router.get("/{project_id}/documents")
async def list_project_documents(
    project_id: UUID, project_repo: ProjectRepoDep, document_repo: DocumentRepoDep
) -> list[DocumentOut]:
    if await project_repo.get_project(project_id) is None:
        raise NotFoundError("project", str(project_id))
    rows = await document_repo.list_documents(project_id=project_id)
    return [DocumentOut.from_row(r) for r in rows]
