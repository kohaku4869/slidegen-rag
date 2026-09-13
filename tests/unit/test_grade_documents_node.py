"""Unit tests for the rewritten grade_documents_node (3-label grading:
relevant / relevant_incomplete / irrelevant) — the drill_candidates
metadata it derives, and that it preserves the old safety-net behavior
(relevant_incomplete content still reaches `documents`, hard floor when
everything is graded irrelevant)."""

from types import SimpleNamespace

import pytest

from slidegen_rag.agent.nodes import grade_documents_node as gdn


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _fake_llm(label_by_doc: dict[str, str]):
    """Builds a stand-in for `_grader_llm` whose `ainvoke` returns the label
    mapped from the HumanMessage content containing the document text."""

    async def ainvoke(messages):
        user_content = messages[-1].content
        for doc, label in label_by_doc.items():
            if doc in user_content:
                return _FakeResponse(label)
        return _FakeResponse("irrelevant")

    return SimpleNamespace(ainvoke=ainvoke)


def _base_state(**overrides) -> dict:
    state = {
        "query": "câu hỏi test",
        "documents": [],
        "document_node_ids": [],
        "pending_documents": [],
        "pending_meta": [],
        "drill_candidates": [],
        "tool_calls": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_relevant_and_incomplete_both_kept_in_documents(monkeypatch):
    monkeypatch.setattr(
        gdn,
        "_grader_llm",
        _fake_llm(
            {
                "full answer chunk": "relevant",
                "summary fragment chunk": "relevant_incomplete",
            }
        ),
    )
    monkeypatch.setattr(
        gdn, "describe_node", lambda node_id: {"has_children": True, "has_parent": True}
    )

    state = _base_state(
        pending_documents=["full answer chunk", "summary fragment chunk"],
        pending_meta=[{"node_id": 10, "layer": 0}, {"node_id": 20, "layer": 1}],
    )

    result = await gdn.grade_documents_node(state)

    assert "full answer chunk" in result["documents"]
    assert "summary fragment chunk" in result["documents"]
    assert result["document_node_ids"] == [10, 20]

    assert len(result["drill_candidates"]) == 1
    assert result["drill_candidates"][0]["node_id"] == 20
    assert result["drill_candidates"][0]["reason"] == "need_detail"


@pytest.mark.asyncio
async def test_incomplete_needs_context_when_no_children(monkeypatch):
    monkeypatch.setattr(
        gdn, "_grader_llm", _fake_llm({"fragment": "relevant_incomplete"})
    )
    monkeypatch.setattr(
        gdn,
        "describe_node",
        lambda node_id: {"has_children": False, "has_parent": True},
    )

    state = _base_state(
        pending_documents=["fragment"],
        pending_meta=[{"node_id": 5, "layer": 0}],
    )

    result = await gdn.grade_documents_node(state)

    assert result["drill_candidates"][0]["reason"] == "need_context"


@pytest.mark.asyncio
async def test_incomplete_isolated_node_not_added_as_candidate(monkeypatch):
    monkeypatch.setattr(
        gdn, "_grader_llm", _fake_llm({"fragment": "relevant_incomplete"})
    )
    monkeypatch.setattr(
        gdn,
        "describe_node",
        lambda node_id: {"has_children": False, "has_parent": False},
    )

    state = _base_state(
        pending_documents=["fragment"],
        pending_meta=[{"node_id": 5, "layer": 0}],
    )

    result = await gdn.grade_documents_node(state)

    # Still kept in documents (safety net), just nothing to drill.
    assert "fragment" in result["documents"]
    assert result["drill_candidates"] == []


@pytest.mark.asyncio
async def test_all_irrelevant_applies_hard_floor(monkeypatch):
    docs = [f"chunk {i}" for i in range(7)]
    monkeypatch.setattr(gdn, "_grader_llm", _fake_llm({d: "irrelevant" for d in docs}))
    monkeypatch.setattr(gdn, "describe_node", lambda node_id: None)

    state = _base_state(
        pending_documents=docs,
        pending_meta=[{"node_id": i, "layer": 0} for i in range(7)],
    )

    result = await gdn.grade_documents_node(state)

    assert len(result["documents"]) == gdn._HARD_FLOOR
    assert result["documents"] == docs[: gdn._HARD_FLOOR]


@pytest.mark.asyncio
async def test_no_pending_keeps_existing_documents_and_prunes_drilled():
    state = _base_state(
        documents=["old doc"],
        document_node_ids=[1],
        drill_candidates=[{"node_id": 1, "content": "old", "reason": "need_detail"}],
        tool_calls=[
            {"name": "expand_node", "args": {"node_id": 1, "direction": "children"}}
        ],
    )

    result = await gdn.grade_documents_node(state)

    assert result["documents"] == ["old doc"]
    # node 1 was just drilled this round — its stale candidate must be pruned.
    assert result["drill_candidates"] == []


@pytest.mark.asyncio
async def test_find_in_subtree_call_prunes_matching_root(monkeypatch):
    monkeypatch.setattr(gdn, "_grader_llm", _fake_llm({}))

    state = _base_state(
        drill_candidates=[{"node_id": 7, "content": "old", "reason": "need_detail"}],
        tool_calls=[
            {"name": "find_in_subtree", "args": {"root_node_id": 7, "patterns": ["x"]}}
        ],
    )

    result = await gdn.grade_documents_node(state)
    assert result["drill_candidates"] == []


@pytest.mark.asyncio
async def test_incomplete_no_children_many_siblings_needs_siblings(monkeypatch):
    monkeypatch.setattr(
        gdn, "_grader_llm", _fake_llm({"fragment": "relevant_incomplete"})
    )
    monkeypatch.setattr(
        gdn,
        "describe_node",
        lambda node_id: {"has_children": False, "has_parent": True, "n_siblings": 4},
    )

    state = _base_state(
        pending_documents=["fragment"],
        pending_meta=[{"node_id": 5, "layer": 0}],
    )

    result = await gdn.grade_documents_node(state)

    assert result["drill_candidates"][0]["reason"] == "need_siblings"


@pytest.mark.asyncio
async def test_incomplete_one_sibling_still_needs_context(monkeypatch):
    monkeypatch.setattr(
        gdn, "_grader_llm", _fake_llm({"fragment": "relevant_incomplete"})
    )
    monkeypatch.setattr(
        gdn,
        "describe_node",
        lambda node_id: {"has_children": False, "has_parent": True, "n_siblings": 1},
    )

    state = _base_state(
        pending_documents=["fragment"],
        pending_meta=[{"node_id": 5, "layer": 0}],
    )

    result = await gdn.grade_documents_node(state)

    assert result["drill_candidates"][0]["reason"] == "need_context"


@pytest.mark.asyncio
async def test_describe_node_none_skips_candidate(monkeypatch):
    monkeypatch.setattr(
        gdn, "_grader_llm", _fake_llm({"fragment": "relevant_incomplete"})
    )
    monkeypatch.setattr(gdn, "describe_node", lambda node_id: None)

    state = _base_state(
        pending_documents=["fragment"],
        pending_meta=[{"node_id": 5, "layer": 0}],
    )

    result = await gdn.grade_documents_node(state)

    assert "fragment" in result["documents"]
    assert result["drill_candidates"] == []
