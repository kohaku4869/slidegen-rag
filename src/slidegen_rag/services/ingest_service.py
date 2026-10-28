"""ingest_service.py — `start_ingest_job`, `finalize_ingest_job`,
`source_md_path`, `index_out_dir`, dùng chung bởi UI và REST API.
`job_registry`/`tree_cache` do caller truyền vào.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from uuid import UUID

from slidegen_rag.services.tree_cache import load_tree
from slidegen_rag.config import ROOT_DIR, SummarizationConfig, TreeConfig
from slidegen_rag.memory.catalog import DocumentRepository
from slidegen_rag.memory.dto import DocumentRow
from slidegen_rag.services.jobs import (
    IngestJob,
    JobRegistry,
    run_ingest_job,
    run_prebuilt_ingest_job,
)

# Demo: upload tài liệu có doc_slug này thì dùng index dựng sẵn, bỏ qua bước
# build (rất lâu). Xoá entry để quay lại ingest thật.
DEMO_PREBUILT_INDEXES: dict[str, Path] = {
    "pldc": ROOT_DIR / "data" / "index" / "pldc_v2.pkl",
}


def source_md_path(project_id: UUID, document_id: UUID) -> Path:
    """Đường dẫn `.md` nguồn tất định theo `document_id` — KHÔNG lệ thuộc
    `doc_slug` (chỉ để hiển thị) nên hai tài liệu trùng tên file gốc trong
    cùng project không bao giờ ghi đè nhau trên đĩa."""
    return (
        ROOT_DIR / "data" / "projects" / str(project_id) / "src" / f"{document_id}.md"
    )


def index_out_dir(project_id: UUID) -> Path:
    return ROOT_DIR / "data" / "projects" / str(project_id) / "index"


async def start_ingest_job(
    document_repo: DocumentRepository,
    *,
    project_id: UUID,
    document: DocumentRow,
    markdown_text: str,
    job_registry: JobRegistry,
) -> IngestJob:
    """Tạo `document_indexes` row (`status='parsing'`) + đăng ký job vào
    `JobRegistry` + submit vào worker thread. Trả về `IngestJob`, để route
    trả `job.job_id` cho client poll qua `GET /jobs/{job_id}`."""
    index_row = await document_repo.start_index(
        document_id=document.id, source_sha1=document.source_sha1
    )

    job = job_registry.new_job(
        document_id=str(document.id),
        index_id=str(index_row.id),
        doc_slug=document.doc_slug,
    )
    src_path = source_md_path(project_id, document.id)
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(markdown_text, encoding="utf-8")

    prebuilt = DEMO_PREBUILT_INDEXES.get(document.doc_slug)
    if prebuilt is not None and prebuilt.exists():
        job_registry.submit(
            job,
            run_prebuilt_ingest_job,
            prebuilt_path=prebuilt,
            out_dir=index_out_dir(project_id),
            doc_id=str(document.id),
            source_path=str(src_path),
        )
        return job

    job_registry.submit(
        job,
        run_ingest_job,
        cfg_tree=TreeConfig(),
        cfg_sum=SummarizationConfig(),
        out_dir=index_out_dir(project_id),
        doc_id=str(document.id),
        source_path=str(src_path),
        markdown_text=markdown_text,
    )
    return job


def _extract_doc_summary_and_outline(tree) -> tuple[str, list[str]]:
    root = next(iter(tree.root_nodes.values()), None)
    doc_summary = root.text if root else ""
    outline_nodes = sorted(
        (n for n in tree.all_nodes.values() if n.depth == 0 and n.title),
        key=lambda n: n.line_start,
    )
    return doc_summary, [n.title for n in outline_nodes]


async def finalize_ingest_job(
    document_repo: DocumentRepository, job: IngestJob, *, tree_cache: OrderedDict
) -> bool:
    """Gọi khi `job.snapshot()["state"]` là `done`/`failed`/`cancelled` —
    commit kết quả vào Postgres (`document_indexes`/`documents`) rồi trả
    `True`. Idempotent: gọi 2 lần với job đã final không sao (finish_index/
    fail_index tự ghi đè cùng giá trị)."""
    snap = job.snapshot()
    if snap["state"] == "done":
        pkl_path = snap["pkl_path"]
        tree = load_tree(tree_cache, pkl_path)
        doc_summary, outline = _extract_doc_summary_and_outline(tree)
        artifact_sha1 = hashlib.sha1(Path(pkl_path).read_bytes()).hexdigest()
        stats_dict = job.stats.to_dict() if job.stats else {}
        await document_repo.finish_index(
            UUID(job.index_id),
            artifact_path=pkl_path,
            artifact_sha1=artifact_sha1,
            build_stats=stats_dict,
            doc_summary=doc_summary,
            outline=outline,
            total_nodes=job.stats.total_nodes if job.stats else len(tree.all_nodes),
            num_layers=job.stats.num_layers if job.stats else tree.num_layers,
            total_tokens=sum(n.n_tokens for n in tree.all_nodes.values()),
        )
        return True
    if snap["state"] in ("failed", "cancelled"):
        await document_repo.fail_index(
            UUID(job.index_id), error_message=snap["error"] or snap["state"]
        )
        return True
    return False
