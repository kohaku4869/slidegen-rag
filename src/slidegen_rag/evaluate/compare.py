"""
So sánh 2 lần chạy (run) của cùng một topic — THUẦN đọc file
(`run_config.json`/`summary.json`), KHÔNG gọi LLM. Xem plan5.md Phần D1.

Dùng cho Tab "🆚 So sánh 2 lần chạy" của dashboard (evaluate/app.py) và có
thể gọi trực tiếp từ CLI/notebook để soát nhanh 2 run.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from slidegen_rag.evaluate import run_config as run_config_mod
from slidegen_rag.evaluate.runs import RunPaths, list_run_ids
from slidegen_rag.evaluate.settings import COVERAGE_NOISE_DELTA, JUDGE_NOISE_DELTA

logger = logging.getLogger(__name__)

# (label hiển thị, key trong summary, ngưỡng nhiễu) — 4 tiêu chí judge +
# total_score dùng JUDGE_NOISE_DELTA (nhiễu judge đo được ±0.06 — xem
# README: "treat any gap below ~0.1 as noise"); coverage/utilization dùng
# COVERAGE_NOISE_DELTA (1.0 điểm phần trăm).
_QUALITY_METRICS = [
    ("Total Score", "total_score", JUDGE_NOISE_DELTA),
    ("Accuracy", "accuracy", JUDGE_NOISE_DELTA),
    ("Completeness", "completeness", JUDGE_NOISE_DELTA),
    ("Relevance", "relevance", JUDGE_NOISE_DELTA),
    ("Faithfulness", "faithfulness", JUDGE_NOISE_DELTA),
    ("Keyword Coverage (strict, %)", "coverage_pct_strict", COVERAGE_NOISE_DELTA),
    ("Answer Coverage (%)", "answer_coverage_pct", COVERAGE_NOISE_DELTA),
    ("Utilization (%)", "utilization_pct", COVERAGE_NOISE_DELTA),
]


def _load_json(path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Không đọc được {path}: {e}")
        return None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    topic: str
    path: RunPaths
    config: Optional[dict]
    summary: Optional[dict]
    has_cost_data: bool
    has_quality_data: bool


def list_runs(topic: str) -> list[RunRecord]:
    """MỚI -> CŨ, bỏ qua thư mục thiếu `summary.json` (không phải run hợp
    lệ — xem `evaluate.runs.list_run_ids`)."""
    records = []
    for run_id in list_run_ids(topic):
        rp = RunPaths.for_run(topic, run_id)
        summary = _load_json(rp.summary)
        if summary is None:
            continue
        cfg = run_config_mod.load_run_config(rp)
        has_quality = summary.get("total_score") is not None
        has_cost = (
            summary.get("total_agent_gemini_tokens") is not None
            or summary.get("session_wallclock_s") is not None
        )
        records.append(
            RunRecord(
                run_id=run_id,
                topic=topic,
                path=rp,
                config=cfg,
                summary=summary,
                has_cost_data=has_cost,
                has_quality_data=has_quality,
            )
        )
    return records


def pick_baseline(topic: str, current_run_id: str) -> tuple[Optional[str], str]:
    """Chọn baseline trong các run CŨ HƠN `current_run_id`: ưu tiên
    `config_fingerprint` trùng ("exact"), không có thì run gần nhất
    ("partial"). Không có run cũ hơn nào -> (None, "none")."""
    records = list_runs(topic)
    ids = [r.run_id for r in records]
    older = records[ids.index(current_run_id) + 1 :] if current_run_id in ids else records
    if not older:
        return None, "none"

    current = next((r for r in records if r.run_id == current_run_id), None)
    current_fp = (current.config or {}).get("config_fingerprint") if current else None
    if current_fp:
        for r in older:
            if (r.config or {}).get("config_fingerprint") == current_fp:
                return r.run_id, "exact"
    return older[0].run_id, "partial"


def config_diff(a: Optional[dict], b: Optional[dict]) -> dict[str, tuple]:
    """{field: (giá_trị_a, giá_trị_b)} cho MỖI field thuộc
    `run_config.FINGERPRINT_FIELDS` mà 2 config khác nhau. `a`/`b` là None
    (thiếu run_config.json) -> mọi field coi là "khác nhau" (không xác nhận
    được có cùng cấu hình hay không)."""
    diff = {}
    for field in run_config_mod.FINGERPRINT_FIELDS:
        va = (a or {}).get(field) if a is not None else "(không có run_config.json)"
        vb = (b or {}).get(field) if b is not None else "(không có run_config.json)"
        if va != vb:
            diff[field] = (va, vb)
    return diff


def _delta_row(label: str, val_a, val_b, noise_delta: float) -> dict:
    if val_a is None or val_b is None:
        return {
            "metric": label,
            "a": val_a,
            "b": val_b,
            "delta": None,
            "delta_label": "không có dữ liệu",
        }
    delta = val_a - val_b
    if abs(delta) < noise_delta:
        delta_label = "≈ nhiễu"
    elif delta > 0:
        delta_label = "tốt hơn"
    else:
        delta_label = "kém hơn"
    return {"metric": label, "a": val_a, "b": val_b, "delta": delta, "delta_label": delta_label}


def _build_quality_rows(summary_a: Optional[dict], summary_b: Optional[dict]) -> list[dict]:
    rows = []
    for label, key, noise in _QUALITY_METRICS:
        val_a = (summary_a or {}).get(key)
        val_b = (summary_b or {}).get(key)
        rows.append(_delta_row(label, val_a, val_b, noise))
    return rows


def _avg_or_none(total, count) -> Optional[float]:
    if total is None or not count:
        return None
    return total / count


def _cost_delta_row(label: str, val_a, val_b) -> dict:
    """Không có ngưỡng nhiễu cho cost/latency (khác `_delta_row`) — mọi
    chênh lệch đo được đều đáng báo cáo, không có khái niệm 'nhiễu đo lường'
    như judge score."""
    if val_a is None or val_b is None:
        return {"metric": label, "a": None, "b": None, "delta": None, "delta_label": "không có dữ liệu"}
    delta = val_a - val_b
    return {
        "metric": label,
        "a": val_a,
        "b": val_b,
        "delta": delta,
        "delta_label": "đắt hơn/chậm hơn" if delta > 0 else ("rẻ hơn/nhanh hơn" if delta < 0 else "bằng nhau"),
    }


def _build_cost_rows(summary_a: Optional[dict], summary_b: Optional[dict]) -> list[dict]:
    a = summary_a or {}
    b = summary_b or {}
    n_a = a.get("total_questions")
    n_b = b.get("total_questions")

    rows = [
        _cost_delta_row(
            "Avg wall time / query (s)",
            _avg_or_none(a.get("total_agent_latency_s"), n_a),
            _avg_or_none(b.get("total_agent_latency_s"), n_b),
        ),
        _cost_delta_row(
            "Avg total tokens / query (Gemini+local)",
            _avg_or_none(
                (a.get("total_agent_gemini_tokens") or 0) + (a.get("total_agent_local_tokens") or 0)
                if a.get("total_agent_gemini_tokens") is not None
                else None,
                n_a,
            ),
            _avg_or_none(
                (b.get("total_agent_gemini_tokens") or 0) + (b.get("total_agent_local_tokens") or 0)
                if b.get("total_agent_gemini_tokens") is not None
                else None,
                n_b,
            ),
        ),
        _cost_delta_row(
            "Avg LLM calls / query (Gemini)",
            _avg_or_none(a.get("total_gemini_calls"), n_a),
            _avg_or_none(b.get("total_gemini_calls"), n_b),
        ),
        _cost_delta_row(
            "Avg LLM calls / query (local)",
            _avg_or_none(a.get("total_local_calls"), n_a),
            _avg_or_none(b.get("total_local_calls"), n_b),
        ),
        _cost_delta_row(
            "Gemini tokens input (tổng run)",
            a.get("total_agent_gemini_input_tokens"),
            b.get("total_agent_gemini_input_tokens"),
        ),
        _cost_delta_row(
            "Gemini tokens output (tổng run)",
            a.get("total_agent_gemini_output_tokens"),
            b.get("total_agent_gemini_output_tokens"),
        ),
        _cost_delta_row(
            "Gemini tokens tổng (tổng run)",
            a.get("total_agent_gemini_tokens"),
            b.get("total_agent_gemini_tokens"),
        ),
        _cost_delta_row("Chi phí Gemini (USD)", a.get("total_cost_usd"), b.get("total_cost_usd")),
        # Local hiện số nhưng KHÔNG có cột USD riêng — GEMINI_PRICING không
        # định giá local, không bịa ra (xem BASELINE_NOTES.md: "miễn phí
        # (self-hosted)").
        _cost_delta_row(
            "Local tokens (tổng run, miễn phí self-hosted)",
            a.get("total_agent_local_tokens"),
            b.get("total_agent_local_tokens"),
        ),
        _cost_delta_row(
            "Memory overhead — Gemini tokens (contextualize+compact)",
            a.get("total_mem_gemini_tokens"),
            b.get("total_mem_gemini_tokens"),
        ),
        _cost_delta_row(
            "Judge tokens (tổng run, KHÔNG cộng vào chi phí/query)",
            a.get("total_judge_tokens"),
            b.get("total_judge_tokens"),
        ),
    ]
    return rows


def compare_runs(topic: str, a: str, b: str) -> dict:
    """So sánh run `a` với run `b` (không bắt buộc thứ tự thời gian — chỉ là
    2 nhãn A/B để hiển thị, dù cách dùng thường gặp là a=hiện tại, b=baseline
    từ `pick_baseline`).

    Returns:
        {"quality_rows": [...], "cost_rows": [...], "config_warnings": [...],
         "data_gaps": [...]}
    """
    rp_a = RunPaths.for_run(topic, a)
    rp_b = RunPaths.for_run(topic, b)
    cfg_a = run_config_mod.load_run_config(rp_a)
    cfg_b = run_config_mod.load_run_config(rp_b)
    summary_a = _load_json(rp_a.summary)
    summary_b = _load_json(rp_b.summary)

    config_warnings = []
    if cfg_a is None or cfg_b is None:
        config_warnings.append(
            "Thiếu run_config.json ở ít nhất 1 run (thường là run đã migrate từ "
            "layout cũ) — KHÔNG xác nhận được 2 run có cùng cấu hình hay không."
        )
    else:
        diff = config_diff(cfg_a, cfg_b)
        if diff:
            fields = ", ".join(diff.keys())
            config_warnings.append(
                f"2 run KHÁC cấu hình ở: {fields} — chênh lệch điểm/chi phí có thể "
                f"đến từ khác cấu hình, không chỉ từ thay đổi đang xét."
            )

    data_gaps = []
    if summary_a is None:
        data_gaps.append(f"Run '{a}' không có summary.json")
    if summary_b is None:
        data_gaps.append(f"Run '{b}' không có summary.json")
    if summary_a is not None and summary_a.get("total_agent_gemini_tokens") is None:
        data_gaps.append(
            f"Run '{a}' không có dữ liệu latency/token (run cũ trước khi tính "
            f"năng đo được thêm vào, hoặc đã migrate từ layout cũ)."
        )
    if summary_b is not None and summary_b.get("total_agent_gemini_tokens") is None:
        data_gaps.append(
            f"Run '{b}' không có dữ liệu latency/token — phần so sánh tiết kiệm "
            f"thời gian/token sẽ trống, chỉ điểm chất lượng so sánh được."
        )

    return {
        "quality_rows": _build_quality_rows(summary_a, summary_b),
        "cost_rows": _build_cost_rows(summary_a, summary_b),
        "config_warnings": config_warnings,
        "data_gaps": data_gaps,
    }
