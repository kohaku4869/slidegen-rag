"""Unit tests cho `ui/jobs.py` — xem plan6.md C3. Không gọi LLM/embedding
thật: `run_ingest_job` nhận `build_ra_fn` injectable trả về một
`RetrievalAugmentation` thật nhưng dùng fake model (giống
tests/unit/test_markdown_spans.py)."""

import tempfile
from pathlib import Path

from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import BaseEmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import BaseSummarizationModel
from slidegen_rag.tree.RetrievalAugmentation import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)
from slidegen_rag.ui.jobs import IngestJob, JobRegistry, run_ingest_job


class FakeEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [float(hash(text) % 100)]

    def create_embeddings(self, texts):
        return [self.create_embedding(t) for t in texts]


class FakeSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return f"tóm tắt: {context[:40]}"

    def batch_summarize(self, contexts, max_tokens=150):
        return [self.summarize(c) for c in contexts]


def _fake_build_ra(cfg_tree, cfg_sum) -> RetrievalAugmentation:
    tb_config = TreeBuilderConfig(
        summarization_model=FakeSummarizationModel(),
        embedding_models={"SBERT": FakeEmbeddingModel()},
    )
    return RetrievalAugmentation(config=RetrievalAugmentationConfig(tree_builder_config=tb_config))


SAMPLE_MD = """\
# CHƯƠNG 1
Đoạn mở đầu.

## 1.1 Mục con
Nội dung mục con.
"""


def _new_job() -> IngestJob:
    return IngestJob(job_id="j1", document_id="d1", index_id="i1", doc_slug="sample")


def test_run_ingest_job_success_writes_pkl_and_stats():
    job = _new_job()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        run_ingest_job(
            job,
            cfg_tree=None,
            cfg_sum=None,
            out_dir=out_dir,
            doc_id="sample",
            source_path="sample.md",
            markdown_text=SAMPLE_MD,
            build_ra_fn=_fake_build_ra,
        )
        snap = job.snapshot()
        assert snap["state"] == "done"
        assert snap["error"] is None
        assert (out_dir / "sample.pkl").exists()
        assert job.stats is not None
        assert job.stats.total_nodes > 0


def test_run_ingest_job_reports_progress_matching_total_units():
    job = _new_job()
    with tempfile.TemporaryDirectory() as tmp:
        run_ingest_job(
            job, cfg_tree=None, cfg_sum=None, out_dir=Path(tmp),
            doc_id="sample", source_path="sample.md", markdown_text=SAMPLE_MD,
            build_ra_fn=_fake_build_ra,
        )
    snap = job.snapshot()
    assert snap["total_units"] > 0
    assert snap["done_units"] == snap["total_units"]


def test_run_ingest_job_cancel_mid_build_marks_cancelled():
    job = _new_job()
    job.cancel_requested = True  # đã yêu cầu huỷ TRƯỚC khi chạy -> raise ở progress đầu tiên
    with tempfile.TemporaryDirectory() as tmp:
        run_ingest_job(
            job, cfg_tree=None, cfg_sum=None, out_dir=Path(tmp),
            doc_id="sample", source_path="sample.md", markdown_text=SAMPLE_MD,
            build_ra_fn=_fake_build_ra,
        )
    snap = job.snapshot()
    assert snap["state"] == "cancelled"
    assert not (Path(tmp) / "sample.pkl").exists()


def test_run_ingest_job_exception_sets_failed_with_traceback():
    job = _new_job()

    def _broken_build_ra(cfg_tree, cfg_sum):
        raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmp:
        run_ingest_job(
            job, cfg_tree=None, cfg_sum=None, out_dir=Path(tmp),
            doc_id="sample", source_path="sample.md", markdown_text=SAMPLE_MD,
            build_ra_fn=_broken_build_ra,
        )
    snap = job.snapshot()
    assert snap["state"] == "failed"
    assert "boom" in snap["error"]
    assert "Traceback" in snap["error"]


def test_job_registry_new_job_and_get_roundtrip():
    registry = JobRegistry()
    job = registry.new_job(document_id="d1", index_id="i1", doc_slug="sample")
    assert registry.get(job.job_id) is job
    assert registry.get("unknown") is None


def test_job_registry_submit_runs_in_background_and_is_pollable():
    registry = JobRegistry()
    job = registry.new_job(document_id="d1", index_id="i1", doc_slug="sample")
    with tempfile.TemporaryDirectory() as tmp:
        registry.submit(
            job, run_ingest_job,
            cfg_tree=None, cfg_sum=None, out_dir=Path(tmp),
            doc_id="sample", source_path="sample.md", markdown_text=SAMPLE_MD,
            build_ra_fn=_fake_build_ra,
        )
        import time
        for _ in range(100):
            if registry.get(job.job_id).snapshot()["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert registry.get(job.job_id).snapshot()["state"] == "done"


def test_job_registry_request_cancel_on_unknown_job_is_noop():
    registry = JobRegistry()
    registry.request_cancel("does-not-exist")  # không raise
