from __future__ import annotations

from fastapi import APIRouter

from slidegen_rag.api.deps import DocumentRepoDep, JobRegistryDep, TreeCacheDep
from slidegen_rag.api.errors import NotFoundError
from slidegen_rag.api.schemas.jobs import JobStatusOut
from slidegen_rag.api.services import ingest_service

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    job_registry: JobRegistryDep,
    document_repo: DocumentRepoDep,
    tree_cache: TreeCacheDep,
) -> JobStatusOut:
    job = job_registry.get(job_id)
    if job is None:
        raise NotFoundError("job", job_id)
    snap = job.snapshot()
    finalized = False
    if snap["state"] in ("done", "failed", "cancelled"):
        finalized = await ingest_service.finalize_ingest_job(
            document_repo, job, tree_cache=tree_cache
        )
    return JobStatusOut(job_id=job_id, finalized=finalized, **snap)
