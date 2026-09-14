"""
Unit tests cho guard chống degenerate output (agent/utils/text_guard.py) —
hàm thuần, không LLM/network.
"""

from slidegen_rag.agent.utils.text_guard import (
    collapse_pathological_repetition,
    has_pathological_repetition,
)


def test_normal_text_is_untouched():
    text = "Trách nhiệm pháp lý là sự phản ứng tiêu cực của nhà nước."
    assert collapse_pathological_repetition(text) == text
    assert has_pathological_repetition(text) is False


def test_markdown_rule_is_not_flagged():
    # Đường kẻ ngang markdown hợp lệ (vài chục ký tự) không được coi là lỗi.
    text = "Phần I\n\n" + "-" * 80 + "\n\nPhần II"
    assert has_pathological_repetition(text) is False
    assert collapse_pathological_repetition(text) == text


def test_long_repeat_run_is_detected_and_collapsed():
    text = "Nội dung hợp lệ." + "-" * 5000 + "Kết thúc."
    assert has_pathological_repetition(text) is True

    out = collapse_pathological_repetition(text)
    assert len(out) < len(text)
    assert "Nội dung hợp lệ." in out
    assert "Kết thúc." in out
    assert "lặp 5000 lần liên tiếp" in out


def test_repeat_of_any_character_is_detected():
    assert has_pathological_repetition("x" * 150) is True
    assert has_pathological_repetition(" " * 150) is True


def test_boundary_99_vs_100_repeats():
    # Ngưỡng là 100 ký tự lặp liên tiếp (1 + \1{99,}).
    assert has_pathological_repetition("a" * 99) is False
    assert has_pathological_repetition("a" * 100) is True


def test_multiple_runs_all_collapsed():
    text = "A" + "=" * 300 + "B" + "#" * 400 + "C"
    out = collapse_pathological_repetition(text)
    assert "lặp 300 lần liên tiếp" in out
    assert "lặp 400 lần liên tiếp" in out
    assert has_pathological_repetition(out) is False


def test_empty_string():
    assert collapse_pathological_repetition("") == ""
    assert has_pathological_repetition("") is False
