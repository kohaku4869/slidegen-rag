"""Unit tests for the drill-down tools added on top of the CRAG agent's
existing tools: `expand_node`'s new `direction` param, `find_in_subtree`
(new tool), and `describe_node` (metadata helper used by grade_documents_node
and ConversationManager). No LLM/network involved."""

from types import SimpleNamespace

from slidegen_rag.agent.tools.tools import (
    describe_node,
    expand_node,
    find_in_subtree,
    set_retrieval_augmentation,
)
from slidegen_rag.tree.structure.tree_structure import Node, Tree


def _make_tree() -> Tree:
    """
    root (0)
     ├─ child1 (1)
     │   ├─ grandchild1 (3)
     │   └─ grandchild2 (4)
     └─ child2 (2)
    """
    nodes = {
        0: Node(text="root summary", index=0, children={1, 2}, embeddings={}),
        1: Node(text="child1 summary", index=1, children={3, 4}, embeddings={}, parent=0),
        2: Node(text="child2 leaf", index=2, children=set(), embeddings={}, parent=0),
        3: Node(text="grandchild1 detail about alpha", index=3, children=set(), embeddings={}, parent=1),
        4: Node(text="grandchild2 detail about beta", index=4, children=set(), embeddings={}, parent=1),
    }
    tree = Tree(
        all_nodes=nodes,
        root_nodes={0: nodes[0]},
        leaf_nodes={2: nodes[2], 3: nodes[3], 4: nodes[4]},
        num_layers=3,
        layer_to_nodes={0: [nodes[3], nodes[4], nodes[2]], 1: [nodes[1]], 2: [nodes[0]]},
        doc_id="doc1",
        parent_of={1: 0, 2: 0, 3: 1, 4: 1},
    )
    return tree


def _setup_ra(tree: Tree, create_embedding=None, context_embedding_model="m"):
    retriever = SimpleNamespace(
        tree_node_index_to_layer={0: 2, 1: 1, 2: 0, 3: 0, 4: 0},
        context_embedding_model=context_embedding_model,
        create_embedding=create_embedding or (lambda text: [0.0, 0.0]),
    )
    ra = SimpleNamespace(tree=tree, retriever=retriever)
    set_retrieval_augmentation(ra)
    return ra


# ── expand_node ──────────────────────────────────────────────────────────


def test_expand_node_children_default_unchanged():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 1})
    assert "grandchild1 detail about alpha" in result
    assert "grandchild2 detail about beta" in result


def test_expand_node_children_explicit():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 1, "direction": "children"})
    assert "grandchild1 detail about alpha" in result


def test_expand_node_leaf_has_no_children():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 2, "direction": "children"})
    assert "leaf node" in result
    assert "child2 leaf" in result


def test_expand_node_parent():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 3, "direction": "parent"})
    assert "child1 summary" in result


def test_expand_node_parent_on_root_soft_fails():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 0, "direction": "parent"})
    assert result.startswith("Node has no parent")


def test_expand_node_siblings():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 3, "direction": "siblings"})
    assert "grandchild2 detail about beta" in result
    assert "grandchild1 detail about alpha" not in result


def test_expand_node_siblings_only_child_soft_fails():
    only_child = Node(text="lonely only child", index=1, children=set(), embeddings={}, parent=0)
    root = Node(text="root", index=0, children={1}, embeddings={})
    nodes = {0: root, 1: only_child}
    tree = Tree(
        all_nodes=nodes,
        root_nodes={0: root},
        leaf_nodes={1: only_child},
        num_layers=2,
        layer_to_nodes={0: [only_child], 1: [root]},
        doc_id="doc_only_child",
        parent_of={1: 0},
    )
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 1, "direction": "siblings"})
    assert result.startswith("Node has no siblings")


def test_expand_node_not_found():
    tree = _make_tree()
    _setup_ra(tree)

    result = expand_node.invoke({"node_id": 999})
    assert result == "Node not found: 999"


# ── find_in_subtree ──────────────────────────────────────────────────────


def test_find_in_subtree_keyword_scopes_to_branch():
    tree = _make_tree()
    _setup_ra(tree)

    result = find_in_subtree.invoke({"root_node_id": 1, "patterns": ["alpha"]})
    assert "grandchild1 detail about alpha" in result
    assert "grandchild2" not in result


def test_find_in_subtree_keyword_does_not_search_outside_root():
    tree = _make_tree()
    _setup_ra(tree)

    # "beta" only exists under node 1's subtree; searching under node 2
    # (a leaf with no children) must not find it.
    result = find_in_subtree.invoke({"root_node_id": 2, "patterns": ["beta"]})
    assert result.startswith("No children to search")


def test_find_in_subtree_no_match():
    tree = _make_tree()
    _setup_ra(tree)

    result = find_in_subtree.invoke({"root_node_id": 1, "patterns": ["nonexistent_term_zzz"]})
    assert result.startswith("No matches found in subtree")


def test_find_in_subtree_dedup_across_patterns():
    tree = _make_tree()
    _setup_ra(tree)

    result = find_in_subtree.invoke({"root_node_id": 1, "patterns": ["detail", "detail"]})
    # "detail" appears in both grandchild texts — same node must not repeat
    # just because it matched twice across pattern iterations run separately.
    assert result.count("grandchild1 detail about alpha") == 1


def test_find_in_subtree_semantic_mode_ranks_by_similarity():
    tree = _make_tree()
    tree.all_nodes[3].embeddings = {"m": [1.0, 0.0]}
    tree.all_nodes[4].embeddings = {"m": [0.0, 1.0]}

    _setup_ra(tree, create_embedding=lambda text: [1.0, 0.0])

    result = find_in_subtree.invoke({
        "root_node_id": 1,
        "patterns": ["alpha-ish query"],
        "mode": "semantic",
        "max_matches_per_pattern": 1,
    })
    assert "grandchild1 detail about alpha" in result
    assert "grandchild2" not in result


def test_find_in_subtree_root_not_found():
    tree = _make_tree()
    _setup_ra(tree)

    result = find_in_subtree.invoke({"root_node_id": 999, "patterns": ["alpha"]})
    assert result == "Node not found: 999"


# ── describe_node ────────────────────────────────────────────────────────


def test_describe_node_reports_children_and_parent():
    tree = _make_tree()
    _setup_ra(tree)

    info = describe_node(1)
    assert info["has_children"] is True
    assert info["has_parent"] is True


def test_describe_node_leaf_no_children():
    tree = _make_tree()
    _setup_ra(tree)

    info = describe_node(3)
    assert info["has_children"] is False
    assert info["has_parent"] is True


def test_describe_node_root_no_parent():
    tree = _make_tree()
    _setup_ra(tree)

    info = describe_node(0)
    assert info["has_parent"] is False


def test_describe_node_missing_node_returns_none():
    tree = _make_tree()
    _setup_ra(tree)

    assert describe_node(999) is None
