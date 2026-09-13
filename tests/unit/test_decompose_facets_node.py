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


def _fake_llm_sequence(contents: list[str]):
    """Returns (llm, calls) — calls["n"] counts ainvoke invocations; each call
    consumes the next content in `contents` (the last one repeats if called
    more times than provided)."""
    calls = {"n": 0}

    async def ainvoke(messages):
        idx = min(calls["n"], len(contents) - 1)
        calls["n"] += 1
        return _FakeResponse(contents[idx])

    return SimpleNamespace(ainvoke=ainvoke), calls


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
async def test_force_split_retries_once_and_accepts_multi_facet_result(monkeypatch):
    query = "So sánh trách nhiệm dân sự và trách nhiệm hình sự của chủ thể vi phạm"
    second_content = (
        "Trách nhiệm dân sự của chủ thể vi phạm là gì?\n"
        "Trách nhiệm hình sự của chủ thể vi phạm là gì?"
    )
    llm, calls = _fake_llm_sequence(
        [
            "Trách nhiệm dân sự và hình sự của chủ thể vi phạm là gì?",
            second_content,
        ]
    )
    monkeypatch.setattr(dfn, "_decompose_llm", llm)

    result = await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 2
    assert len(result["facet_queries"]) == 2


@pytest.mark.asyncio
async def test_force_split_retry_still_one_facet_keeps_original(monkeypatch):
    query = "So sánh trách nhiệm dân sự và trách nhiệm hình sự của chủ thể vi phạm"
    llm, calls = _fake_llm_sequence(
        ["một dòng diễn giải lại không tách được thành 2 phần"]
    )
    monkeypatch.setattr(dfn, "_decompose_llm", llm)

    result = await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 2
    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_no_comparison_marker_does_not_retry(monkeypatch):
    query = "Điều kiện áp dụng biện pháp khẩn cấp tạm thời trong tố tụng dân sự là gì?"
    llm, calls = _fake_llm_sequence(["diễn giải lại không có marker so sánh nào cả"])
    monkeypatch.setattr(dfn, "_decompose_llm", llm)

    result = await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 1
    assert len(result["facet_queries"]) == 1


@pytest.mark.asyncio
async def test_already_multi_facet_on_first_try_does_not_retry(monkeypatch):
    query = "So sánh trách nhiệm dân sự và trách nhiệm hình sự của chủ thể vi phạm"
    content = (
        "Trách nhiệm dân sự của chủ thể vi phạm là gì?\n"
        "Trách nhiệm hình sự của chủ thể vi phạm là gì?"
    )
    llm, calls = _fake_llm_sequence([content])
    monkeypatch.setattr(dfn, "_decompose_llm", llm)

    result = await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 1
    assert len(result["facet_queries"]) == 2


@pytest.mark.asyncio
async def test_force_split_retry_raises_keeps_original_single_facet(monkeypatch):
    query = "So sánh trách nhiệm dân sự và trách nhiệm hình sự của chủ thể vi phạm"
    calls = {"n": 0}

    async def ainvoke(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("một dòng diễn giải lại")
        raise RuntimeError("boom")

    monkeypatch.setattr(dfn, "_decompose_llm", SimpleNamespace(ainvoke=ainvoke))

    result = await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 2
    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_short_query_with_marker_does_not_retry(monkeypatch):
    query = "phân biệt A và B"  # < 8 từ
    llm, calls = _fake_llm_sequence(["A và B khác nhau ở chỗ nào đó"])
    monkeypatch.setattr(dfn, "_decompose_llm", llm)

    await dfn.decompose_facets_node({"query": query})

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_single_facet_never_empty(monkeypatch):
    query = "Câu hỏi bất kỳ"
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm_raises(RuntimeError("boom")))

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query
