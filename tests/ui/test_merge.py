"""Unit tests cho `ui/merge.py::merge_trees` — xem plan6.md A2 + Verification
#8. Cây test là các `Tree`/`Node` tổng hợp nhỏ (không cần LLM/embedding thật),
đủ để kiểm bất biến offset/parent_of/id duy nhất."""

from itertools import chain

import numpy as np
import pytest

from slidegen_rag.tree.structure.tree_structure import Node, Tree
from slidegen_rag.ui.merge import (
    NODE_SLOT_SIZE,
    merge_trees,
    offset_for_ref_no,
    resolve_node,
)


def _make_tree(doc_id: str, seed: float) -> Tree:
    """Cây tổng hợp: 2 lá + 1 root, id gốc 0/1/2, có embeddings SBERT."""
    leaf0 = Node(
        f"{doc_id} leaf 0", 0, set(), {"SBERT": [seed, 0.1, 0.2]},
        doc_id=doc_id, layer=0, kind="leaf",
    )
    leaf1 = Node(
        f"{doc_id} leaf 1", 1, set(), {"SBERT": [seed + 0.05, 0.1, 0.2]},
        doc_id=doc_id, layer=0, kind="leaf",
    )
    root = Node(
        f"{doc_id} root", 2, {0, 1}, {"SBERT": [seed + 0.02, 0.15, 0.25]},
        doc_id=doc_id, layer=1, kind="root",
    )
    leaf0.parent = 2
    leaf1.parent = 2
    all_nodes = {0: leaf0, 1: leaf1, 2: root}
    layer_to_nodes = {0: [leaf0, leaf1], 1: [root]}
    parent_of = {0: 2, 1: 2}
    return Tree(
        all_nodes=all_nodes,
        root_nodes={2: root},
        leaf_nodes={0: leaf0, 1: leaf1},
        num_layers=2,
        layer_to_nodes=layer_to_nodes,
        schema_version=2,
        doc_id=doc_id,
        parent_of=parent_of,
    )


def test_offsets_are_deterministic_per_ref_no():
    assert offset_for_ref_no(1) == 0
    assert offset_for_ref_no(2) == NODE_SLOT_SIZE
    assert offset_for_ref_no(5) == 4 * NODE_SLOT_SIZE


def test_merge_two_docs_produces_unique_ids_and_expected_offsets():
    tree_a = _make_tree("docA", 0.0)
    tree_b = _make_tree("docB", 1.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a), (2, "b", "Doc B", tree_b)])

    assert len(merged.all_nodes) == 6
    assert set(merged.all_nodes) == {0, 1, 2, NODE_SLOT_SIZE, NODE_SLOT_SIZE + 1, NODE_SLOT_SIZE + 2}


def test_merge_invariant_all_nodes_matches_layer_to_nodes():
    tree_a = _make_tree("docA", 0.0)
    tree_b = _make_tree("docB", 1.0)
    tree_c = _make_tree("docC", 2.0)
    merged = merge_trees(
        [(1, "a", "Doc A", tree_a), (2, "b", "Doc B", tree_b), (3, "c", "Doc C", tree_c)]
    )
    layer_ids = {n.index for n in chain.from_iterable(merged.layer_to_nodes.values())}
    assert set(merged.all_nodes) == layer_ids


def test_merge_preserves_parent_of_after_offset():
    tree_a = _make_tree("docA", 0.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a)])
    for child, parent in merged.parent_of.items():
        assert merged.all_nodes[child].parent == parent
        assert child in merged.all_nodes[parent].children


def test_merge_does_not_mutate_source_tree():
    tree_a = _make_tree("docA", 0.0)
    original_children = {i: set(n.children) for i, n in tree_a.all_nodes.items()}
    merge_trees([(1, "a", "Doc A", tree_a), (2, "b", "Doc B", _make_tree("docB", 1.0))])
    for i, n in tree_a.all_nodes.items():
        assert n.children == original_children[i]
        assert n.index == i  # index gốc không bị dịch


def test_merge_shares_embeddings_by_reference_not_copy():
    tree_a = _make_tree("docA", 0.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a)])
    assert merged.all_nodes[0].embeddings is tree_a.all_nodes[0].embeddings


def test_resolve_node_roundtrip_for_every_merged_id():
    tree_a = _make_tree("docA", 0.0)
    tree_b = _make_tree("docB", 1.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a), (2, "b", "Doc B", tree_b)])

    for merged_id in merged.all_nodes:
        result = resolve_node(merged, merged_id)
        assert result is not None
        doc_slice, orig = result
        assert doc_slice.offset + orig == merged_id
        expected_key = "a" if merged_id < NODE_SLOT_SIZE else "b"
        assert doc_slice.doc_key == expected_key


def test_resolve_node_returns_none_for_unknown_id():
    tree_a = _make_tree("docA", 0.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a)])
    assert resolve_node(merged, 999) is None


def test_offset_unchanged_when_document_dropped_from_selection():
    """Bất biến quan trọng nhất của A2: offset của một tài liệu phụ thuộc
    THUẦN TUÝ vào ref_no, không phụ thuộc tập tài liệu đang được chọn để gộp."""
    tree_a = _make_tree("docA", 0.0)
    tree_b = _make_tree("docB", 1.0)
    tree_c = _make_tree("docC", 2.0)

    merged_all = merge_trees(
        [(1, "a", "Doc A", tree_a), (2, "b", "Doc B", tree_b), (3, "c", "Doc C", tree_c)]
    )
    offset_c_full = next(s.offset for s in merged_all.doc_slices if s.doc_key == "c")

    # Bỏ docB khỏi tập chọn giữa chừng.
    merged_without_b = merge_trees([(1, "a", "Doc A", tree_a), (3, "c", "Doc C", tree_c)])
    offset_c_partial = next(s.offset for s in merged_without_b.doc_slices if s.doc_key == "c")

    assert offset_c_full == offset_c_partial == offset_for_ref_no(3)


def test_single_document_merge_is_n_equals_1_path():
    tree_a = _make_tree("docA", 0.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a)])
    assert len(merged.doc_slices) == 1
    assert set(merged.all_nodes) == {0, 1, 2}


def test_rejects_document_exceeding_node_slot_size(monkeypatch):
    import slidegen_rag.ui.merge as merge_mod

    monkeypatch.setattr(merge_mod, "NODE_SLOT_SIZE", 2)
    tree_a = _make_tree("docA", 0.0)  # 3 nodes > slot size 2
    with pytest.raises(ValueError):
        merge_mod.merge_trees([(1, "a", "Doc A", tree_a)])


def test_emb_matrix_attached_and_l2_normalized():
    tree_a = _make_tree("docA", 0.0)
    tree_b = _make_tree("docB", 1.0)
    merged = merge_trees([(1, "a", "Doc A", tree_a), (2, "b", "Doc B", tree_b)])

    matrix = merged._emb_matrix
    assert matrix.shape == (6, 3)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert merged._emb_node_order == sorted(merged.all_nodes)
