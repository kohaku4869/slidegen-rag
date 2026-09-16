"""evaluate/compare.py — thuần đọc file, không gọi LLM. Xem plan5.md D1."""

import json

import pytest

from slidegen_rag.evaluate import compare
from slidegen_rag.evaluate.runs import RunPaths


@pytest.fixture(autouse=True)
def _patch_data_test_dir(tmp_path, monkeypatch):
    from slidegen_rag.evaluate import runs as runs_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    return tmp_path


def _make_run(topic, run_id, *, summary=None, config=None):
    rp = RunPaths.for_run(topic, run_id)
    rp.root.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        rp.summary.write_text(json.dumps(summary), encoding="utf-8")
    if config is not None:
        rp.run_config.write_text(json.dumps(config), encoding="utf-8")
    return rp


_FULL_SUMMARY = {
    "total_score": 8.0,
    "accuracy": 8.0,
    "completeness": 8.0,
    "relevance": 8.0,
    "faithfulness": 8.0,
    "coverage_pct_strict": 80.0,
    "answer_coverage_pct": 70.0,
    "utilization_pct": 85.0,
    "total_questions": 30,
    "total_agent_gemini_tokens": 30000,
    "total_agent_gemini_input_tokens": 25000,
    "total_agent_gemini_output_tokens": 5000,
    "total_agent_local_tokens": 1000,
    "total_gemini_calls": 150,
    "total_local_calls": 30,
    "total_agent_latency_s": 300.0,
    "total_cost_usd": 0.05,
    "total_mem_gemini_tokens": 0,
    "total_judge_tokens": 5000,
    "session_wallclock_s": 320.0,
}

_CFG_FULL = {
    "mode": "full",
    "generate_provider": "gemini",
    "gemini_model": "gemini-3.5-flash-lite",
    "local_base_url": None,
    "index_source_sha1": "abc",
    "web_search_enabled": False,
    "max_turns": None,
    "questions_sha1": "qsha",
    "config_fingerprint": "fp-full-1",
}


def test_list_runs_skips_missing_summary(tmp_path):
    _make_run("t", "20260101_000000", summary=_FULL_SUMMARY)
    RunPaths.for_run("t", "20260102_000000").root.mkdir(parents=True)  # no summary.json

    records = compare.list_runs("t")

    assert [r.run_id for r in records] == ["20260101_000000"]
    assert records[0].has_quality_data is True
    assert records[0].has_cost_data is True


def test_pick_baseline_prefers_exact_fingerprint_match(tmp_path):
    _make_run("t", "20260101_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)
    other_cfg = {**_CFG_FULL, "config_fingerprint": "fp-different"}
    _make_run("t", "20260102_000000", summary=_FULL_SUMMARY, config=other_cfg)
    _make_run("t", "20260103_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)

    baseline_id, label = compare.pick_baseline("t", "20260103_000000")

    assert baseline_id == "20260101_000000"  # cùng fingerprint, dù không phải run gần nhất
    assert label == "exact"


def test_pick_baseline_falls_back_to_most_recent_older_run(tmp_path):
    other_cfg = {**_CFG_FULL, "config_fingerprint": "fp-a"}
    _make_run("t", "20260101_000000", summary=_FULL_SUMMARY, config=other_cfg)
    other_cfg2 = {**_CFG_FULL, "config_fingerprint": "fp-b"}
    _make_run("t", "20260102_000000", summary=_FULL_SUMMARY, config=other_cfg2)
    _make_run("t", "20260103_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)

    baseline_id, label = compare.pick_baseline("t", "20260103_000000")

    assert baseline_id == "20260102_000000"
    assert label == "partial"


def test_pick_baseline_no_older_runs(tmp_path):
    _make_run("t", "20260101_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)

    baseline_id, label = compare.pick_baseline("t", "20260101_000000")

    assert baseline_id is None
    assert label == "none"


def test_config_diff_reports_only_fingerprint_fields():
    a = {**_CFG_FULL, "started_at": "2026-01-01", "generate_provider": "gemini"}
    b = {**_CFG_FULL, "started_at": "2099-01-01", "generate_provider": "local"}

    diff = compare.config_diff(a, b)

    assert "generate_provider" in diff
    assert diff["generate_provider"] == ("gemini", "local")
    assert "started_at" not in diff  # không thuộc FINGERPRINT_FIELDS


def test_quality_rows_flag_noise_vs_real_delta():
    summary_a = {**_FULL_SUMMARY, "total_score": 8.05}
    summary_b = {**_FULL_SUMMARY, "total_score": 8.00}
    rows = compare._build_quality_rows(summary_a, summary_b)
    total_row = next(r for r in rows if r["metric"] == "Total Score")
    assert total_row["delta_label"] == "≈ nhiễu"  # |0.05| < JUDGE_NOISE_DELTA(0.1)

    summary_a2 = {**_FULL_SUMMARY, "total_score": 8.5}
    rows2 = compare._build_quality_rows(summary_a2, summary_b)
    total_row2 = next(r for r in rows2 if r["metric"] == "Total Score")
    assert total_row2["delta_label"] == "tốt hơn"


def test_compare_runs_missing_cost_data_leaves_delta_blank_not_zero(tmp_path):
    """Run nền (migrated) không có token/latency -> delta phải là None
    ('không có dữ liệu'), KHÔNG được hiện 0 hay -100% (xem plan5.md D1)."""
    quality_only_summary = {
        "total_score": 7.5,
        "accuracy": 7.5,
        "completeness": 7.5,
        "relevance": 7.5,
        "faithfulness": 7.5,
        "coverage_pct_strict": 75.0,
        "answer_coverage_pct": 60.0,
        "utilization_pct": 80.0,
        "total_questions": 30,
    }
    migrated_cfg = {
        "mode": "unknown",
        "config_fingerprint": None,
        "migrated": True,
    }
    _make_run("t", "20260101_000000", summary=quality_only_summary, config=migrated_cfg)
    _make_run("t", "20260102_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)

    result = compare.compare_runs("t", "20260102_000000", "20260101_000000")

    cost_rows = {r["metric"]: r for r in result["cost_rows"]}
    wall_time_row = cost_rows["Avg wall time / query (s)"]
    assert wall_time_row["b"] is None
    assert wall_time_row["delta"] is None
    assert wall_time_row["delta_label"] == "không có dữ liệu"

    gemini_tokens_row = cost_rows["Gemini tokens tổng (tổng run)"]
    assert gemini_tokens_row["delta"] is None

    quality_rows = {r["metric"]: r for r in result["quality_rows"]}
    # Quality VẪN so sánh được dù thiếu cost data.
    assert quality_rows["Total Score"]["a"] == 8.0
    assert quality_rows["Total Score"]["b"] == 7.5

    assert any("dữ liệu latency/token" in gap for gap in result["data_gaps"])
    assert any("run_config.json" in w or "KHÁC cấu hình" in w for w in result["config_warnings"])


def test_compare_runs_missing_summary_entirely(tmp_path):
    _make_run("t", "20260101_000000", summary=_FULL_SUMMARY, config=_CFG_FULL)
    RunPaths.for_run("t", "20260102_000000").root.mkdir(parents=True)  # không có summary.json

    result = compare.compare_runs("t", "20260102_000000", "20260101_000000")

    assert any("20260102_000000" in gap for gap in result["data_gaps"])
    for row in result["quality_rows"]:
        assert row["a"] is None
        assert row["delta_label"] == "không có dữ liệu"
