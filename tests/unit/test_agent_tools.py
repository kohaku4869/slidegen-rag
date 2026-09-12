"""Unit tests for slidegen_rag.agent.tools.tools BM25 caching & lifecycle."""

from types import SimpleNamespace

from slidegen_rag.agent.tools.tools import (
    _get_bm25,
    clear_bm25_cache,
    keyword_search,
    set_retrieval_augmentation,
)
from slidegen_rag.tree.structure.tree_structure import Node, Tree


def _make_dummy_tree(doc_id: str, texts: list[str]) -> Tree:
    nodes = {
        i: Node(text=text, index=i, children=set(), embeddings={})
        for i, text in enumerate(texts)
    }
    return Tree(
        all_nodes=nodes,
        root_nodes=nodes,
        leaf_nodes=nodes,
        num_layers=1,
        layer_to_nodes={0: list(nodes.values())},
        doc_id=doc_id,
    )


def test_get_bm25_caches_directly_on_tree():
    tree = _make_dummy_tree("doc1", ["hợp đồng dân sự", "quyền sở hữu tài sản"])
    ra = SimpleNamespace(tree=tree)

    assert not hasattr(tree, "_bm25_cache")

    bm25_first, nodes_first = _get_bm25(ra)
    assert bm25_first is not None
    assert len(nodes_first) == 2
    assert hasattr(tree, "_bm25_cache")

    # Second call must return the exact same instance from tree._bm25_cache
    bm25_second, nodes_second = _get_bm25(ra)
    assert bm25_first is bm25_second
    assert nodes_first is nodes_second


def test_get_bm25_no_collision_across_trees():
    tree_a = _make_dummy_tree(
        "docA",
        [
            "bộ luật dân sự",
            "thừa kế theo pháp luật",
            "tố tụng hình sự",
            "nghị định chính phủ",
        ],
    )
    tree_b = _make_dummy_tree(
        "docB",
        [
            "kế toán tài chính",
            "báo cáo lưu chuyển tiền tệ",
            "kiểm toán độc lập",
            "chuẩn mực kế toán",
        ],
    )

    ra_a = SimpleNamespace(tree=tree_a)
    ra_b = SimpleNamespace(tree=tree_b)

    bm25_a, _ = _get_bm25(ra_a)
    bm25_b, _ = _get_bm25(ra_b)

    assert bm25_a is not bm25_b
    assert tree_a._bm25_cache[0] is bm25_a
    assert tree_b._bm25_cache[0] is bm25_b

    # Verifying searches run against the correct tree
    scores_a = bm25_a.get_scores(["thừa", "kế"])
    scores_b = bm25_b.get_scores(["thừa", "kế"])
    assert max(scores_a) > 0
    assert max(scores_b) == 0.0


def test_clear_bm25_cache():
    tree = _make_dummy_tree("doc1", ["điều khoản thi hành"])
    ra = SimpleNamespace(tree=tree)

    _get_bm25(ra)
    assert hasattr(tree, "_bm25_cache")

    clear_bm25_cache(tree)
    assert not hasattr(tree, "_bm25_cache")


def test_get_bm25_handles_none_and_empty():
    ra_empty = SimpleNamespace(tree=None)
    bm25, nodes = _get_bm25(ra_empty)
    assert bm25 is None
    assert nodes is None

    tree_no_text = _make_dummy_tree("empty", ["", "   "])
    ra_no_text = SimpleNamespace(tree=tree_no_text)
    bm25_empty, nodes_empty = _get_bm25(ra_no_text)
    assert bm25_empty is None
    assert nodes_empty == []


def test_keyword_search_uses_tree_cache():
    tree = _make_dummy_tree(
        "doc_search",
        [
            "Hiến pháp năm 2013",
            "Luật đất đai sửa đổi",
            "Nghị định hướng dẫn thi hành",
            "Thông tư liên tịch",
        ],
    )
    ra = SimpleNamespace(
        tree=tree,
        retriever=SimpleNamespace(tree_node_index_to_layer={0: 0, 1: 0, 2: 0, 3: 0}),
    )
    set_retrieval_augmentation(ra)

    result = keyword_search.invoke({"query": "Hiến pháp"})
    assert "Hiến pháp năm 2013" in result
    assert hasattr(tree, "_bm25_cache")
