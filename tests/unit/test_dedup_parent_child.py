"""Unit tests for tree.utils.dedup_parent_child — pure logic, no LLM/network."""

from tree.structure.tree_structure import Node
from tree.utils import dedup_parent_child


def _node(index: int, children=None) -> Node:
    return Node(text=f"node-{index}", index=index, children=set(children or []), embeddings={})


def test_parent_dropped_when_child_present():
    parent = _node(1, children={2, 3})
    child = _node(2)
    result = dedup_parent_child([parent, child])
    assert result == [child]


def test_parent_kept_when_no_child_present():
    parent = _node(1, children={2, 3})
    unrelated = _node(4)
    result = dedup_parent_child([parent, unrelated])
    assert result == [parent, unrelated]


def test_relative_order_preserved():
    a = _node(1)
    parent = _node(2, children={3})
    child = _node(3)
    b = _node(4)
    result = dedup_parent_child([a, parent, child, b])
    assert [n.index for n in result] == [1, 3, 4]


def test_leaf_with_no_children_never_dropped():
    leaf = _node(5, children=set())
    assert dedup_parent_child([leaf]) == [leaf]


def test_empty_input():
    assert dedup_parent_child([]) == []
