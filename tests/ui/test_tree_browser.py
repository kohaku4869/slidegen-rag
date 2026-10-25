"""Unit tests cho phần thuần (không Streamlit) của
`ui/components/tree_browser.py` — xem plan6.md D2."""

from slidegen_rag.tree.structure.tree_structure import Node, Tree
from ui.components.tree_browser import _describe_node, _expand_path_to


def _make_tree() -> Tree:
    leaf0 = Node("lá 0", 0, set(), {}, layer=0, depth=1, parent=2)
    leaf1 = Node("lá 1", 1, set(), {}, layer=0, depth=1, parent=2)
    root = Node("gốc", 2, {0, 1}, {}, layer=1, depth=0, kind="root")
    all_nodes = {0: leaf0, 1: leaf1, 2: root}
    return Tree(
        all_nodes=all_nodes, root_nodes={2: root}, leaf_nodes={0: leaf0, 1: leaf1},
        num_layers=2, layer_to_nodes={0: [leaf0, leaf1], 1: [root]},
        schema_version=2, parent_of={0: 2, 1: 2},
    )


def test_describe_node_root_has_children_no_parent():
    tree = _make_tree()
    info = _describe_node(tree, 2)
    assert info["has_children"] is True
    assert info["has_parent"] is False
    assert info["n_children"] == 2


def test_describe_node_leaf_counts_siblings():
    tree = _make_tree()
    info = _describe_node(tree, 0)
    assert info["has_children"] is False
    assert info["has_parent"] is True
    assert info["n_siblings"] == 1  # leaf1 là anh em


def test_describe_node_missing_returns_none():
    tree = _make_tree()
    assert _describe_node(tree, 999) is None


def test_expand_path_to_walks_up_to_root():
    tree = _make_tree()
    expanded: set[int] = set()
    _expand_path_to(tree, 0, expanded)
    assert expanded == {0, 2}


def test_expand_path_to_does_not_infinite_loop_on_cycle():
    tree = _make_tree()
    tree.all_nodes[2].parent = 0  # cạnh giả tạo vòng lặp
    expanded: set[int] = set()
    _expand_path_to(tree, 0, expanded)
    assert expanded  # kết thúc, không treo
