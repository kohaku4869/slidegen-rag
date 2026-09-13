"""Unit tests for decompose_facets_node: single-topic fast-path (sub-question
MUST be the original query verbatim), multi-facet parsing (one plain question
per line — no LLM-assigned LABEL, see plan2.md FIX C), MAX_FACETS cap, and
fail-closed fallback (parse error / LLM error -> exactly 1 facet, never
0 facets)."""

from types import SimpleNamespace

import pytest

from slidegen_rag.agent.nodes import decompose_facets_node as dfn


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _fake_llm(content: str):
    async def ainvoke(messages):
        return _FakeResponse(content)

    return SimpleNamespace(ainvoke=ainvoke)


def _fake_llm_raises(exc: Exception):
    async def ainvoke(messages):
        raise exc

    return SimpleNamespace(ainvoke=ainvoke)


@pytest.mark.asyncio
async def test_single_topic_query_forces_exactly_one_facet_with_verbatim_query(
    monkeypatch,
):
    query = "Quy phạm pháp luật là gì?"
    # Even if the LLM tries to paraphrase the sub-question, code must override
    # it back to the original query verbatim for the single-facet fast-path.
    monkeypatch.setattr(
        dfn, "_decompose_llm", _fake_llm("Khái niệm quy phạm pháp luật")
    )

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query
    assert result["facet_queries"][0]["label"].startswith("1. ")


@pytest.mark.asyncio
async def test_multi_facet_query_parses_each_line(monkeypatch):
    content = (
        "Chủ thể trong quan hệ pháp luật dân sự là gì?\n"
        "Chủ thể trong quan hệ pháp luật hình sự là gì?\n"
        "Chủ thể trong quan hệ pháp luật hành chính là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node(
        {
            "query": "So sánh chủ thể trong quan hệ pháp luật dân sự, hình sự và hành chính"
        }
    )

    facets = result["facet_queries"]
    assert len(facets) == 3
    assert facets[0]["query"] == "Chủ thể trong quan hệ pháp luật dân sự là gì?"
    assert facets[1]["query"] == "Chủ thể trong quan hệ pháp luật hình sự là gì?"
    assert facets[2]["query"] == "Chủ thể trong quan hệ pháp luật hành chính là gì?"
    # Labels are code-generated (index prefix), not parsed from the LLM.
    assert facets[0]["label"].startswith("1. ")
    assert facets[1]["label"].startswith("2. ")
    assert facets[2]["label"].startswith("3. ")


@pytest.mark.asyncio
async def test_more_than_max_facets_is_truncated(monkeypatch):
    lines = [f"Sub-question number {i} about something specific" for i in range(1, 7)]
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm("\n".join(lines)))

    result = await dfn.decompose_facets_node({"query": "some compound query"})

    assert len(result["facet_queries"]) == dfn.MAX_FACETS
    assert result["facet_queries"][0]["query"] == lines[0]
    assert result["facet_queries"][-1]["query"] == lines[dfn.MAX_FACETS - 1]


@pytest.mark.asyncio
async def test_single_line_output_forces_verbatim_original_query(monkeypatch):
    query = "Câu hỏi không rõ ràng"
    monkeypatch.setattr(
        dfn, "_decompose_llm", _fake_llm("gibberish with no pipe separator")
    )

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_empty_llm_output_falls_back_to_single_facet(monkeypatch):
    query = "Câu hỏi bất kỳ"
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm("\n\n   \n"))

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_preamble_line_is_ignored(monkeypatch):
    content = (
        "Các khía cạnh:\n"
        "Chủ thể trong quan hệ pháp luật dân sự là gì?\n"
        "Chủ thể trong quan hệ pháp luật hình sự là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node({"query": "so sánh dân sự và hình sự"})

    facets = result["facet_queries"]
    assert len(facets) == 2
    assert facets[0]["query"] == "Chủ thể trong quan hệ pháp luật dân sự là gì?"
    assert facets[1]["query"] == "Chủ thể trong quan hệ pháp luật hình sự là gì?"


@pytest.mark.asyncio
async def test_bullets_and_numbering_are_stripped(monkeypatch):
    content = (
        "1. Chủ thể trong quan hệ pháp luật dân sự là gì?\n"
        "- Chủ thể trong quan hệ pháp luật hình sự là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node({"query": "so sánh dân sự và hình sự"})

    facets = result["facet_queries"]
    assert len(facets) == 2
    assert facets[0]["query"] == "Chủ thể trong quan hệ pháp luật dân sự là gì?"
    assert facets[1]["query"] == "Chủ thể trong quan hệ pháp luật hình sự là gì?"


@pytest.mark.asyncio
async def test_legacy_pipe_format_still_parses_subquestion(monkeypatch):
    content = (
        "Dân sự|Chủ thể trong quan hệ pháp luật dân sự là gì?\n"
        "Hình sự|Chủ thể trong quan hệ pháp luật hình sự là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node({"query": "so sánh dân sự và hình sự"})

    facets = result["facet_queries"]
    assert len(facets) == 2
    assert facets[0]["query"] == "Chủ thể trong quan hệ pháp luật dân sự là gì?"
    assert facets[1]["query"] == "Chủ thể trong quan hệ pháp luật hình sự là gì?"
    # Label is code-generated from the sub-question, not the old LLM label.
    assert facets[0]["label"].startswith("1. ")


@pytest.mark.asyncio
async def test_labels_are_unique_even_for_identical_subquestions(monkeypatch):
    content = (
        "Điều kiện áp dụng biện pháp khẩn cấp tạm thời là gì?\n"
        "Điều kiện áp dụng biện pháp khẩn cấp tạm thời là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node({"query": "so sánh 2 trường hợp"})

    facets = result["facet_queries"]
    assert len(facets) == 2
    assert facets[0]["label"] != facets[1]["label"]


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_single_facet_never_empty(monkeypatch):
    query = "Câu hỏi bất kỳ"
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm_raises(RuntimeError("boom")))

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query
