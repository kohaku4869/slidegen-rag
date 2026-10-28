"""
jobs.py — `JobRegistry` + `run_ingest_job()`: ingest nền có thanh tiến độ
thật (xem plan6.md C3). Dùng chung bởi UI Streamlit và REST API.

Thread riêng, KHÔNG subprocess: `ChatNVIDIASummarizationModel.batch_summarize`
gọi `asyncio.run(...)` bên trong — trên một thread sạch (không phải runtime
loop của `ui/runtime.py`) điều này chạy đúng. `ThreadPoolExecutor(max_workers=1)`
tách biệt hẳn với `AgentRuntime` — ingest và agent-chat không tranh nhau CPU/
GIL cho LLM call, và một job ingest luôn chạy xong trước job kế tiếp (tuần tự
theo project, tránh 2 job cùng ghi `document_indexes` chồng lấn `attempt_no`).

`run_ingest_job()` CHỦ ĐỘNG KHÔNG đụng Postgres — nó chỉ là hàm thuần
markdown -> (.pkl, BuildStats, error). Trang gọi (project_home.py/document.py)
poll `job.snapshot()` qua `st.fragment(run_every=1.0)`, và khi `state` là
`done`/`failed` thì tự gọi `catalog.finish_index()`/`fail_index()` qua
`AgentRuntime.run(...)` — giữ mọi việc ghi DB async trên MỘT loop (xem A1),
không phải trên worker thread của job này.

**Không dùng `RetrievalAugmentation.add_documents()`** — nó có một lệnh
`input()` chặn khi cây đã tồn tại, sẽ treo worker vĩnh viễn (xem
`tree/RetrievalAugmentation.py:162-167`). Luôn gọi `build_from_text` trực
tiếp như `ingest.ingest_file()` làm.
"""

from __future__ import annotations

import pickle
import random
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slidegen_rag.tree.stats import BuildStats


class IngestCancelled(Exception):
    """Raise từ `progress_callback` khi `job.cancel_requested` — hạt cancel
    nhỏ nhất là MỘT layer (LLM call của layer đang chạy vẫn tốn tiền tới hết,
    không có resume — xem plan6.md C3)."""


@dataclass
class IngestJob:
    job_id: str
    document_id: str
    index_id: str
    doc_slug: str
    state: str = "queued"  # queued | running | done | failed | cancelled
    total_units: int = 0
    done_units: int = 0
    current_label: str = ""
    error: str | None = None
    stats: BuildStats | None = None
    pkl_path: str | None = None
    cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "total_units": self.total_units,
                "done_units": self.done_units,
                "current_label": self.current_label,
                "error": self.error,
                "pkl_path": self.pkl_path,
            }


class JobRegistry:
    """Process-wide (xem `get_job_registry`), qua mọi rerun Streamlit và mọi
    tab."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ingest-worker"
        )

    def new_job(self, *, document_id: str, index_id: str, doc_slug: str) -> IngestJob:
        job = IngestJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            index_id=index_id,
            doc_slug=doc_slug,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: IngestJob, fn, *args: Any, **kwargs: Any) -> None:
        self._executor.submit(fn, job, *args, **kwargs)

    def request_cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is not None:
            with job._lock:
                job.cancel_requested = True


_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def get_job_registry() -> JobRegistry:
    """Singleton process-wide cho UI Streamlit (API giữ registry riêng ở
    `app.state.job_registry`)."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry


def run_prebuilt_ingest_job(
    job: IngestJob,
    *,
    prebuilt_path: Path,
    out_dir: Path,
    doc_id: str,
    source_path: str,
    simulated_seconds: tuple[float, float] = (30.0, 40.0),
) -> None:
    """Đường tắt cho demo: thay vì build lại cây, nạp `.pkl` dựng sẵn rồi gán
    lại `doc_id`/`source_path` của document hiện tại và lưu ra `out_dir` —
    cùng hợp đồng đầu ra với `run_ingest_job` (`job.pkl_path`, `job.stats`)."""
    from slidegen_rag.tree import legacy_pickle  # noqa: F401 — shim unpickle

    with job._lock:
        job.state = "running"

    try:
        # Demo: kéo dài thanh tiến độ cho giống một lần index thật.
        total_steps = 100
        step_delay = random.uniform(*simulated_seconds) / total_steps
        for step in range(total_steps):
            with job._lock:
                if job.cancel_requested:
                    raise IngestCancelled("Huỷ theo yêu cầu người dùng")
                job.current_label = "Đang index tài liệu"
                job.done_units = step
                job.total_units = total_steps
            time.sleep(step_delay)

        with open(prebuilt_path, "rb") as f:
            tree = pickle.load(f)

        tree.doc_id = doc_id
        tree.source_path = source_path
        for node in tree.all_nodes.values():
            node.doc_id = doc_id

        pkl_path = out_dir / f"{doc_id}.pkl"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(tree, f)

        with job._lock:
            job.state = "done"
            job.stats = getattr(tree, "build_stats", None)
            job.pkl_path = str(pkl_path)
            job.total_units = job.done_units = total_steps

    except IngestCancelled as exc:
        with job._lock:
            job.state = "cancelled"
            job.error = str(exc)

    except Exception:
        with job._lock:
            job.state = "failed"
            job.error = traceback.format_exc()


def run_ingest_job(
    job: IngestJob,
    *,
    cfg_tree,
    cfg_sum,
    out_dir: Path,
    doc_id: str,
    source_path: str,
    markdown_text: str,
    build_ra_fn=None,
) -> None:
    """Chạy trong worker thread của `JobRegistry`. Không nhận `RetrieverConfig`
    riêng vì `catalog.py`/UI chỉ cần `.pkl` + `BuildStats` — `TreeRetriever`
    thật được tạo lại khi merge tree lúc chat (xem `ui/merge.py`, `ui/data.py`).

    `build_ra_fn` mặc định `ingest._build_ra` (models thật, cần network/API
    key) — tham số tồn tại chỉ để test thay bằng fake model, không đổi hành
    vi production."""
    if build_ra_fn is None:
        from slidegen_rag.ingest import _build_ra as build_ra_fn

    def _progress(label: str, done: int, total: int) -> None:
        with job._lock:
            if job.cancel_requested:
                raise IngestCancelled("Huỷ theo yêu cầu người dùng")
            job.current_label = label
            job.done_units = done
            job.total_units = total

    with job._lock:
        job.state = "running"

    try:
        ra = build_ra_fn(cfg_tree, cfg_sum)
        ra.tree = ra.tree_builder.build_from_text(
            markdown_text=markdown_text,
            doc_id=doc_id,
            source_path=source_path,
            progress_callback=_progress,
        )

        pkl_path = out_dir / f"{doc_id}.pkl"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        ra.save(str(pkl_path))

        with job._lock:
            job.state = "done"
            job.stats = ra.tree.build_stats
            job.pkl_path = str(pkl_path)

    except IngestCancelled as exc:
        with job._lock:
            job.state = "cancelled"
            job.error = str(exc)

    except Exception:
        with job._lock:
            job.state = "failed"
            job.error = traceback.format_exc()
