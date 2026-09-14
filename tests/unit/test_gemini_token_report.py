"""Unit tests for scripts/gemini_token_report.py's parsing logic: sums
Gemini token/cost across every '=== CÂU HỎI N ===' block in a saved
trajectory.txt, supporting both the new 'Tokens Gemini:' line and the old
combined 'Tokens:' line (see agent/observability/trajectory.py)."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "gemini_token_report.py"
)
_spec = importlib.util.spec_from_file_location("gemini_token_report", _SCRIPT_PATH)
gemini_token_report = importlib.util.module_from_spec(_spec)
sys.modules["gemini_token_report"] = gemini_token_report
_spec.loader.exec_module(gemini_token_report)

parse_trajectory_file = gemini_token_report.parse_trajectory_file
format_report = gemini_token_report.format_report


def test_parses_new_gemini_specific_format():
    text = (
        "=== CÂU HỎI 1 ===\n\n"
        "Câu hỏi : abc\n"
        "Tokens Gemini: in=1,000 out=500 tổng=1,500 | Chi phí ước tính: $0.001300 USD (~33.02 VNĐ)\n"
        "\n\n"
        "=== CÂU HỎI 2 ===\n\n"
        "Câu hỏi : def\n"
        "Tokens Gemini: in=2,000 out=800 tổng=2,800 | Chi phí ước tính: $0.002600 USD (~66.04 VNĐ)\n"
    )
    rows = parse_trajectory_file(text)

    assert len(rows) == 2
    assert rows[0] == {
        "num": 1,
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "cost_usd": 0.001300,
        "legacy": False,
    }
    assert rows[1]["num"] == 2
    assert rows[1]["total_tokens"] == 2800


def test_falls_back_to_legacy_format_and_flags_it():
    text = (
        "=== CÂU HỎI 1 ===\n\n"
        "Câu hỏi : abc\n"
        "Tokens: in=14,068 out=2,058 tổng=16,126 | Chi phí ước tính: $0.009365 USD (~237.88 VNĐ)\n"
    )
    rows = parse_trajectory_file(text)

    assert len(rows) == 1
    assert rows[0]["legacy"] is True
    assert rows[0]["total_tokens"] == 16126
    assert rows[0]["cost_usd"] == 0.009365


def test_question_without_tokens_line_is_skipped():
    text = (
        "=== CÂU HỎI 1 ===\n\n"
        "Câu hỏi : abc\n"
        "LỖI AGENT: something broke\n"
        "\n\n"
        "=== CÂU HỎI 2 ===\n\n"
        "Tokens Gemini: in=100 out=50 tổng=150 | Chi phí ước tính: $0.000100 USD (~2.54 VNĐ)\n"
    )
    rows = parse_trajectory_file(text)

    assert len(rows) == 1
    assert rows[0]["num"] == 2


def test_format_report_sums_totals_and_flags_legacy_count():
    rows = [
        {
            "num": 1,
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "cost_usd": 0.001,
            "legacy": False,
        },
        {
            "num": 2,
            "input_tokens": 2000,
            "output_tokens": 800,
            "total_tokens": 2800,
            "cost_usd": 0.002,
            "legacy": True,
        },
    ]
    report = format_report(Path("dummy.txt"), rows)

    assert "Input tokens : 3,000" in report
    assert "Output tokens: 1,300" in report
    assert "Tổng tokens  : 4,300" in report
    assert "1 câu dùng format 'Tokens:' cũ" in report
