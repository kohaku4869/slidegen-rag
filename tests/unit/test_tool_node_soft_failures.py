"""Unit tests: 3 soft-failure prefixes that were missing
from `SOFT_FAILURE_PREFIXES` — dead-end `expand_node` outputs ("no children",
"no parent content", "no siblings content") must be filtered out of
`_extract_chunks` and reported as `soft_failed=True` in search_history,
instead of leaking into pending_documents and getting graded/counted as
progress."""

import asyncio

import pytest

from slidegen_rag.agent.nodes import tool_node as tool_node_module
from slidegen_rag.agent.nodes.tool_node import _extract_chunks, tool_node


@pytest.mark.parametrize(
    "raw",
    [
        "Node has no children (leaf node): 42",
        "No parent content found for node 42.",
        "No siblings content found for node 42.",
    ],
)
def test_extract_chunks_drops_new_soft_failure_strings(raw):
    assert _extract_chunks(raw, "expand_node") == []


def test_extract_chunks_regression_two_real_nodes_still_two_chunks():
    raw = "§N1|L0§ " + "a" * 30 + "\n\n§N2|L0§ " + "b" * 30
    chunks = _extract_chunks(raw, "expand_node")
    assert len(chunks) == 2
    assert chunks[0]["node_id"] == 1
    assert chunks[1]["node_id"] == 2


def test_tool_node_dead_end_output_yields_no_pending_and_marks_soft_failed(monkeypatch):
    class _StubTool:
        def invoke(self, args, config=None):
            return "Node has no children (leaf node): 42"

    monkeypatch.setitem(tool_node_module.TOOL_MAP, "expand_node", _StubTool())

    state = {
        "tool_calls": [
            {"name": "expand_node", "args": {"node_id": 42, "direction": "children"}}
        ],
        "seen_fingerprints": [],
        "search_history": [],
        "retrieve_count": 0,
    }

    result = asyncio.run(tool_node(state, config={}))

    assert result["pending_documents"] == []
    assert result["search_history"][0]["soft_failed"] is True


def test_tool_node_real_output_still_populates_pending(monkeypatch):
    class _StubTool:
        def invoke(self, args, config=None):
            return (
                "§N1|L0§ "
                + ("real content " * 3)
                + "\n\n"
                + "§N2|L0§ "
                + ("more content " * 3)
            )

    monkeypatch.setitem(tool_node_module.TOOL_MAP, "expand_node", _StubTool())

    state = {
        "tool_calls": [
            {"name": "expand_node", "args": {"node_id": 1, "direction": "children"}}
        ],
        "seen_fingerprints": [],
        "search_history": [],
        "retrieve_count": 0,
    }

    result = asyncio.run(tool_node(state, config={}))

    assert len(result["pending_documents"]) == 2
    assert result["search_history"][0]["soft_failed"] is False
