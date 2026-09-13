"""Unit tests for FIX A (plan2.md): `retrieve_node._digest_hint` picks the
right tool/direction from a node_digest entry's child/sibling counts, and
`_format_drill_candidates` renders the new "need_siblings" reason."""

from slidegen_rag.agent.nodes.retrieve_node import (
    _digest_hint,
    _format_drill_candidates,
)


def _entry(**overrides):
    base = {
        "id": 1,
        "layer": 0,
        "snippet": "snippet",
        "has_children": False,
        "heading": "",
    }
    base.update(overrides)
    return base


def test_digest_hint_unknown_when_n_children_none():
    assert _digest_hint(_entry(n_children=None, n_siblings=None, has_parent=None)) == ""


def test_digest_hint_many_children_suggests_find_in_subtree():
    hint = _digest_hint(_entry(n_children=5, n_siblings=0, has_parent=True))
    assert "find_in_subtree" in hint
    assert "root_node_id=1" in hint


def test_digest_hint_some_children_suggests_expand_children():
    hint = _digest_hint(_entry(n_children=2, n_siblings=0, has_parent=True))
    assert hint == 'expand_node(node_id=1, direction="children")'


def test_digest_hint_no_children_many_siblings_suggests_expand_siblings():
    hint = _digest_hint(_entry(n_children=0, n_siblings=2, has_parent=True))
    assert hint == 'expand_node(node_id=1, direction="siblings")'


def test_digest_hint_no_children_few_siblings_suggests_expand_parent():
    hint = _digest_hint(_entry(n_children=0, n_siblings=1, has_parent=True))
    assert hint == 'expand_node(node_id=1, direction="parent")'


def test_digest_hint_leaf_root_has_no_suggestion():
    hint = _digest_hint(_entry(n_children=0, n_siblings=0, has_parent=False))
    assert hint == ""


def test_format_drill_candidates_need_siblings_renders_siblings_direction():
    candidates = [{"node_id": 7, "reason": "need_siblings", "content": "some content"}]
    result = _format_drill_candidates(candidates)
    assert 'expand_node(node_id=7, direction="siblings")' in result
