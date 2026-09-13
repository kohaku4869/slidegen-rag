"""Unit tests for decompose_facets_node: single-topic fast-path (sub-question
MUST be the original query verbatim), multi-facet parsing, MAX_FACETS cap,
and fail-closed fallback (parse error / LLM error -> exactly 1 facet, never
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
        dfn, "_decompose_llm", _fake_llm("Định nghĩa|Khái niệm quy phạm pháp luật")
    )

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_multi_facet_query_parses_each_line(monkeypatch):
    content = (
        "Chủ thể dân sự|Chủ thể trong quan hệ pháp luật dân sự là gì?\n"
        "Chủ thể hình sự|Chủ thể trong quan hệ pháp luật hình sự là gì?\n"
        "Chủ thể hành chính|Chủ thể trong quan hệ pháp luật hành chính là gì?"
    )
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm(content))

    result = await dfn.decompose_facets_node(
        {
            "query": "So sánh chủ thể trong quan hệ pháp luật dân sự, hình sự và hành chính"
        }
    )

    facets = result["facet_queries"]
    assert len(facets) == 3
    assert facets[0] == {
        "label": "Chủ thể dân sự",
        "query": "Chủ thể trong quan hệ pháp luật dân sự là gì?",
    }
    assert facets[1]["label"] == "Chủ thể hình sự"
    assert facets[2]["label"] == "Chủ thể hành chính"


@pytest.mark.asyncio
async def test_more_than_max_facets_is_truncated(monkeypatch):
    lines = [f"Facet {i}|Sub-question {i}" for i in range(1, 7)]  # 6 lines
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm("\n".join(lines)))

    result = await dfn.decompose_facets_node({"query": "some compound query"})

    assert len(result["facet_queries"]) == dfn.MAX_FACETS
    assert result["facet_queries"][0]["label"] == "Facet 1"
    assert result["facet_queries"][-1]["label"] == f"Facet {dfn.MAX_FACETS}"


@pytest.mark.asyncio
async def test_unparseable_output_falls_back_to_single_facet(monkeypatch):
    query = "Câu hỏi không rõ ràng"
    monkeypatch.setattr(
        dfn, "_decompose_llm", _fake_llm("gibberish with no pipe separator")
    )

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_single_facet_never_empty(monkeypatch):
    query = "Câu hỏi bất kỳ"
    monkeypatch.setattr(dfn, "_decompose_llm", _fake_llm_raises(RuntimeError("boom")))

    result = await dfn.decompose_facets_node({"query": query})

    assert len(result["facet_queries"]) == 1
    assert result["facet_queries"][0]["query"] == query
