"""Unit tests cho `ui/steps.py::render_step` — xem plan6.md C4."""

from slidegen_rag.agent.observability.trajectory import LLMCallRecord, NodeStep
from ui.steps import render_step


def test_retrieve_step_lists_tool_calls():
    step = NodeStep(
        index=0,
        name="retrieve",
        duration_s=1.23,
        llm_calls=[
            LLMCallRecord(
                model="gemini", provider="google", duration_s=1.0,
                tool_calls_emitted=["semantic_search", "keyword_search"],
            )
        ],
    )
    text = render_step(step)
    assert "`retrieve`" in text
    assert "1.23s" in text
    assert "semantic_search" in text and "keyword_search" in text


def test_grade_documents_step_shows_retained_count():
    # `update_summary` đã qua `_summarize_update()` (trajectory.py) — list bị
    # rút thành len(list) trước khi tới đây, nên "documents" LÀ int, không
    # phải list (regression test cho bug len(int) đã gặp khi chạy thật).
    step = NodeStep(
        index=1, name="grade_documents", duration_s=0.5,
        update_summary={"documents": 3},
    )
    assert "3" in render_step(step)


def test_context_evaluator_step_shows_sufficiency_and_feedback():
    step = NodeStep(
        index=2, name="context_evaluator", duration_s=0.2,
        update_summary={"context_sufficient": False, "feedback": "thiếu số liệu"},
    )
    text = render_step(step)
    assert "❌" in text
    assert "thiếu số liệu" in text


def test_context_evaluator_step_sufficient_has_no_dangling_dash():
    step = NodeStep(
        index=3, name="context_evaluator", duration_s=0.2,
        update_summary={"context_sufficient": True, "feedback": None},
    )
    text = render_step(step)
    assert "✅" in text
    assert text.strip().endswith("đủ ngữ cảnh")


def test_generate_step_has_fixed_label():
    step = NodeStep(index=4, name="generate", duration_s=3.4)
    assert "Đang soạn câu trả lời" in render_step(step)


def test_facet_prefix_shown_when_present():
    step = NodeStep(index=5, name="retrieve", duration_s=0.1, facet="giá cả")
    assert render_step(step).startswith("**[giá cả]**")


def test_unknown_step_name_has_no_detail_line():
    step = NodeStep(index=6, name="decompose_facets", duration_s=0.1)
    text = render_step(step)
    assert "\n" not in text
    assert "`decompose_facets`" in text
