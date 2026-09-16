"""
merge.py — Gộp nhiều cây RAPTOR (`.pkl`, mỗi cây một tài liệu) thành MỘT cây
trong bộ nhớ mà `RetrievalAugmentation`/agent dùng chung (xem plan6.md A2).

Đã kiểm chứng: tầng agent chỉ đụng `all_nodes`, `children`, `parent`,
`parent_of`, `retriever.tree_node_index_to_layer` — không đọc `root_nodes`
hay `source_text`, nên gộp cây rẻ hơn nhiều so với viết lại `tree_retriever`
+ 5 tool bằng pgvector.

Bất biến quan trọng nhất: offset của một tài liệu là hàm THUẦN TUÝ của
`ref_no` (thứ tự trong project), KHÔNG phụ thuộc tập tài liệu đang được chọn
để gộp. Nếu offset phụ thuộc tập chọn thì bỏ bớt một nguồn giữa chừng sẽ làm
`cited_node_ids_last_turn` của lượt trước trỏ nhầm node ở lượt sau —
`seed_drill_candidates()` sẽ đào sai chỗ mà không báo lỗi.
"""

from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from slidegen_rag.config import RetrieverConfig
from slidegen_rag.tree.structure.tree_structure import Node, Tree

# Slot cố định dành cho mỗi tài liệu, đủ dư so với ~320 node/tài liệu thực tế
# (xem plan6.md C2). Node index của tài liệu ref_no=k nằm trong
# [ (k-1)*NODE_SLOT_SIZE , k*NODE_SLOT_SIZE ) sau khi offset — dict-based
# storage của Tree.all_nodes nên độ thưa này không tốn bộ nhớ thật.
NODE_SLOT_SIZE = 1_000_000


@dataclass(frozen=True)
class DocSlice:
    doc_key: str
    doc_id: str
    title: str
    offset: int
    n_nodes: int


def offset_for_ref_no(ref_no: int) -> int:
    """Offset tất định của tài liệu theo `ref_no` — xem docstring module."""
    return (ref_no - 1) * NODE_SLOT_SIZE


def merge_trees(parts: list[tuple[int, str, str, Tree]]) -> Tree:
    """Gộp nhiều cây thành một `Tree` mới, không mutate cây gốc.

    Args:
        parts: list các `(ref_no, doc_key, title, tree)`. `doc_key` là định
            danh tài liệu phía UI (thường là `str(document_id)`); `tree` là
            cây gốc đã load từ `.pkl` (nằm trong `st.cache_resource`, DÙNG
            CHUNG nhiều lượt — không được sửa tại chỗ).

    Trả về `Tree` mới với hai thuộc tính thêm: `doc_slices` (list[DocSlice],
    sắp theo offset tăng dần) và `slice_offsets` (list[int] song song, dùng
    cho `resolve_node`). Node được `copy.copy()` (chia sẻ lại `embeddings`
    theo tham chiếu — không nhân đôi vector) rồi dịch `index`/`children`/
    `parent` theo offset tất định.
    """
    ordered = sorted(parts, key=lambda p: p[0])

    all_nodes: dict[int, Node] = {}
    layer_to_nodes: dict[int, list[Node]] = {}
    leaf_nodes: dict[int, Node] = {}
    root_nodes: dict[int, Node] = {}
    parent_of: dict[int, int] = {}
    doc_slices: list[DocSlice] = []
    slice_offsets: list[int] = []

    for ref_no, doc_key, title, tree in ordered:
        offset = offset_for_ref_no(ref_no)
        n_nodes = len(tree.all_nodes)
        if n_nodes >= NODE_SLOT_SIZE:
            raise ValueError(
                f"Tài liệu {doc_key!r} có {n_nodes} node, vượt NODE_SLOT_SIZE="
                f"{NODE_SLOT_SIZE} — tăng NODE_SLOT_SIZE trước khi gộp."
            )
        doc_slices.append(
            DocSlice(
                doc_key=doc_key,
                doc_id=tree.doc_id,
                title=title,
                offset=offset,
                n_nodes=n_nodes,
            )
        )
        slice_offsets.append(offset)

        for old_idx, node in tree.all_nodes.items():
            new_idx = old_idx + offset
            new_node = copy.copy(node)
            new_node.index = new_idx
            new_node.children = {c + offset for c in node.children}
            new_node.parent = None if node.parent is None else node.parent + offset
            all_nodes[new_idx] = new_node
            layer_to_nodes.setdefault(node.layer, []).append(new_node)

        for old_idx in tree.leaf_nodes:
            leaf_nodes[old_idx + offset] = all_nodes[old_idx + offset]
        for old_idx in tree.root_nodes:
            root_nodes[old_idx + offset] = all_nodes[old_idx + offset]
        for child_old, parent_old in tree.parent_of.items():
            parent_of[child_old + offset] = parent_old + offset

    num_layers = max(layer_to_nodes.keys(), default=-1) + 1

    if set(all_nodes) != {n.index for nodes in layer_to_nodes.values() for n in nodes}:
        raise AssertionError(
            "merge_trees: all_nodes lệch với layer_to_nodes sau khi offset"
        )

    merged = Tree(
        all_nodes=all_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_nodes,
        num_layers=num_layers,
        layer_to_nodes=layer_to_nodes,
        build_stats=None,
        schema_version=2,
        doc_id="+".join(d.doc_key for d in doc_slices),
        source_path="",
        source_sha1="",
        # Rỗng có chủ đích — map ngược qua resolve_node() rồi đọc source_text
        # của tree GỐC (nằm trong cache_resource), xem D1.
        source_text="",
        parent_of=parent_of,
    )
    merged.doc_slices = doc_slices
    merged.slice_offsets = slice_offsets
    _attach_embedding_matrix(merged)
    return merged


def resolve_node(merged: Tree, merged_node_id: int) -> tuple[DocSlice, int] | None:
    """Map ngược: node id trong cây gộp -> (DocSlice, node id gốc trong cây
    của riêng tài liệu đó). None nếu id không thuộc slice nào (đã bị xoá
    khỏi tập chọn hoặc không hợp lệ)."""
    slices: list[DocSlice] = getattr(merged, "doc_slices", None) or []
    offsets: list[int] = getattr(merged, "slice_offsets", None) or []
    if not slices:
        return None
    i = bisect_right(offsets, merged_node_id) - 1
    if i < 0:
        return None
    doc_slice = slices[i]
    orig = merged_node_id - doc_slice.offset
    if orig < 0 or orig >= doc_slice.n_nodes:
        return None
    return doc_slice, orig


def _attach_embedding_matrix(tree: Tree) -> None:
    """Cache `tree._emb_matrix` (np.float32, đã L2-normalize) + thứ tự node
    tương ứng `tree._emb_node_order` — fast-path vectorized cho
    `TreeRetriever.retrieve_information_collapse_tree` (xem
    `tree/retriever/tree_retriever.py`). 10 tài liệu ≈ 3.200 node thì tính
    cosine distance bằng list comprehension gọi scipy từng node một (code cũ)
    chậm thấy rõ; nhân ma trận numpy 1 lần thay thế toàn bộ vòng lặp đó."""
    model_key = RetrieverConfig.CONTEXT_EMBEDDING_MODEL
    node_list = [tree.all_nodes[i] for i in sorted(tree.all_nodes)]
    vectors = []
    for node in node_list:
        emb = node.embeddings.get(model_key) if isinstance(node.embeddings, dict) else None
        if emb is None:
            return  # thiếu embedding ở model_key hiện tại -> bỏ fast-path, fallback code cũ
        vectors.append(emb)
    if not vectors:
        return
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tree._emb_matrix = matrix / norms
    tree._emb_node_order = [n.index for n in node_list]
