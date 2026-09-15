"""Unit tests: `tool_node._build_digest` calls
`describe_node()` for real tree metadata (n_children/n_siblings/has_parent/
heading) instead of guessing `has_children` from layer, and falls back to
the old heuristic only when `describe_node` can't resolve the node."""

from slidegen_rag.agent.nodes import tool_node as tool_node_module
from slidegen_rag.agent.nodes.tool_node import _build_digest


def _chunk_info(node_id, layer, text="some retrieved text here"):
    return {"node_id": node_id, "layer": layer, "text": text}


def test_build_digest_uses_describe_node_metadata(monkeypatch):
    monkeypatch.setattr(
        tool_node_module,
        "describe_node",
        lambda node_id: {
            "has_children": True,
            "has_parent": True,
            "heading_path": ("Chương I", "Điều 5"),
            "layer": 1,
            "n_children": 3,
            "n_siblings": 2,
        },
    )

    digest = _build_digest([_chunk_info(10, 1)])

    assert len(digest) == 1
    entry = digest[0]
    assert entry["n_children"] == 3
    assert entry["n_siblings"] == 2
    assert entry["has_parent"] is True
    assert entry["heading"] == "Chương I > Điều 5"


def test_build_digest_falls_back_to_layer_heuristic_when_describe_node_returns_none(
    monkeypatch,
):
    monkeypatch.setattr(tool_node_module, "describe_node", lambda node_id: None)

    digest = _build_digest([_chunk_info(10, 1)])

    assert len(digest) == 1
    entry = digest[0]
    assert entry["has_children"] is True  # layer=1 > 0 → heuristic fallback
    assert entry["n_children"] is None
    assert entry["n_siblings"] is None
    assert entry["has_parent"] is None
    assert entry["heading"] == ""


def test_build_digest_skips_chunks_without_node_id(monkeypatch):
    monkeypatch.setattr(tool_node_module, "describe_node", lambda node_id: None)

    digest = _build_digest([_chunk_info(None, 1)])

    assert digest == []
