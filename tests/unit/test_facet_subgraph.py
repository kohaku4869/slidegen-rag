"""Unit tests for build_facet_subgraph(): with the 4 inner nodes mocked out,
confirm the subgraph exits correctly at END either when context_sufficient
becomes True, or once the facet's own MAX_RETRIEVE budget is exhausted
(best-effort)."""

import pytest

from slidegen_rag.agent.graph import facet_subgraph as fs
from slidegen_rag.agent.graph import router
from slidegen_rag.agent.graph.facet_state import make_initial_facet_state


async def _fake_retrieve(state):
    rc = state.get("retrieve_count", 0)
    return {
        "tool_calls": [{"name": "semantic_search", "args": {"query": "x"}}],
        "retrieve_count": rc + 1,
    }


async def _fake_tool(state):
    return {
        "pending_documents": ["a fresh chunk"],
        "seen_fingerprints": [],
        "search_history": [
            {
                "round": state.get("retrieve_count", 0),
                "tool": "semantic_search",
                "query": "x",
                "new_chunks": 1,
                "soft_failed": False,
            }
        ],
        "node_digest": [],
        "pending_meta": [{"node_id": None, "layer": None}],
    }


async def _fake_grade(state):
    return {
        "documents": state.get("documents", []) + state.get("pending_documents", []),
        "document_node_ids": state.get("document_node_ids", []) + [None],
        "drill_candidates": [],
    }


def _make_fake_evaluator(sufficient_after: int):
    async def _fake_eval(state):
        rc = state.get("retrieve_count", 0)
        suff = rc >= sufficient_after
        return {
            "context_sufficient": suff,
            "feedback": "" if suff else "need more",
            "missing_aspects": [] if suff else ["missing thing"],
        }

    return _fake_eval


@pytest.mark.asyncio
async def test_exits_at_end_when_context_becomes_sufficient(monkeypatch):
    monkeypatch.setattr(fs, "retrieve_node", _fake_retrieve)
    monkeypatch.setattr(fs, "tool_node", _fake_tool)
    monkeypatch.setattr(fs, "grade_documents_node", _fake_grade)
    monkeypatch.setattr(
        fs, "context_evaluator_node", _make_fake_evaluator(sufficient_after=1)
    )

    graph = fs.build_facet_subgraph()
    result = await graph.ainvoke(make_initial_facet_state("Facet A", "sub question"))

    assert result["context_sufficient"] is True
    assert result["retrieve_count"] == 1


@pytest.mark.asyncio
async def test_exits_at_end_when_max_retrieve_exhausted(monkeypatch):
    monkeypatch.setattr(router, "MAX_RETRIEVE", 2)
    monkeypatch.setattr(fs, "retrieve_node", _fake_retrieve)
    monkeypatch.setattr(fs, "tool_node", _fake_tool)
    monkeypatch.setattr(fs, "grade_documents_node", _fake_grade)
    monkeypatch.setattr(
        fs, "context_evaluator_node", _make_fake_evaluator(sufficient_after=999)
    )

    graph = fs.build_facet_subgraph()
    result = await graph.ainvoke(make_initial_facet_state("Facet B", "sub question"))

    assert result["context_sufficient"] is False
    assert result["retrieve_count"] == 2
