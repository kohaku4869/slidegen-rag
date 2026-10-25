"""
components/tree_browser.py — Hai tab "Cây"/"Bảng" cho tab Cây của Document
detail (xem plan6.md D2). Tự viết, không component bên thứ ba.

**Không dùng `agent/tools/tools.py::describe_node`**: nó đọc từ `_ra` toàn
cục (tree đang set cho lượt chat hiện tại) — sai ngữ cảnh ở đây, browser này
luôn phải mô tả đúng CÂY ĐANG TRUYỀN VÀO (một tài liệu cụ thể ở trang
document.py), bất kể `_ra` đang trỏ tới cây gộp nào cho phiên chat khác.
`_describe_node` bên dưới tính lại y hệt logic của `describe_node` nhưng
nhận `tree` tường minh.
"""

from __future__ import annotations

import streamlit as st

TREE_TAB_NODE_LIMIT = 2000


def _describe_node(tree, node_id: int) -> dict | None:
    node = tree.all_nodes.get(node_id)
    if node is None:
        return None
    parent_of_map = getattr(tree, "parent_of", None) or {}
    parent_id = node.parent if node.parent is not None else parent_of_map.get(node_id)
    parent = tree.all_nodes.get(parent_id) if parent_id is not None else None
    sibling_count = max(len(getattr(parent, "children", None) or ()) - 1, 0) if parent else 0
    return {
        "has_children": bool(node.children),
        "has_parent": parent_id is not None,
        "heading_path": node.heading_path or (),
        "layer": node.layer,
        "n_children": len(node.children),
        "n_siblings": sibling_count,
    }


def _expand_path_to(tree, node_id: int, expanded: set[int]) -> None:
    parent_of_map = getattr(tree, "parent_of", None) or {}
    current = node_id
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        expanded.add(current)
        node = tree.all_nodes.get(current)
        parent = node.parent if node and node.parent is not None else parent_of_map.get(current)
        current = parent


def render_tree_browser(tree, *, key_prefix: str, focus_node_id: int | None = None) -> None:
    n_nodes = len(tree.all_nodes)
    expanded_key = f"{key_prefix}__expanded"
    selected_key = f"{key_prefix}__selected"

    if expanded_key not in st.session_state:
        st.session_state[expanded_key] = set(tree.root_nodes.keys())
    if focus_node_id is not None and focus_node_id in tree.all_nodes:
        _expand_path_to(tree, focus_node_id, st.session_state[expanded_key])
        st.session_state[selected_key] = focus_node_id

    if n_nodes > TREE_TAB_NODE_LIMIT:
        (tab_table,) = st.tabs(["Bảng"])
        st.info(
            f"Cây có {n_nodes:,} node (> {TREE_TAB_NODE_LIMIT:,}) — ẩn tab Cây, "
            "dùng Bảng để tìm kiếm/sắp xếp."
        )
        with tab_table:
            _render_table_tab(tree, selected_key)
    else:
        tab_tree, tab_table = st.tabs(["Cây", "Bảng"])
        with tab_tree:
            _render_tree_tab(tree, expanded_key, selected_key)
        with tab_table:
            _render_table_tab(tree, selected_key)

    selected_id = st.session_state.get(selected_key)
    if selected_id is not None and selected_id in tree.all_nodes:
        st.divider()
        _render_node_detail(tree, selected_id)


def _render_tree_tab(tree, expanded_key: str, selected_key: str) -> None:
    expanded: set[int] = st.session_state[expanded_key]

    def _walk(node_id: int) -> None:
        node = tree.all_nodes.get(node_id)
        if node is None:
            return
        is_open = node_id in expanded
        col_toggle, col_label = st.columns([0.06, 0.94])
        with col_toggle:
            if node.children:
                if st.button("▾" if is_open else "▸", key=f"{expanded_key}_toggle_{node_id}"):
                    expanded.discard(node_id) if is_open else expanded.add(node_id)
                    st.rerun()
        with col_label:
            indent = "    " * node.depth
            title = node.title or (node.text[:60] if node.text else f"Node {node.index}")
            if st.button(f"{indent}{title}", key=f"{expanded_key}_select_{node_id}"):
                st.session_state[selected_key] = node_id
        if is_open:
            for child_id in sorted(node.children):
                _walk(child_id)

    for root_id in sorted(tree.root_nodes.keys()):
        _walk(root_id)


def _render_table_tab(tree, selected_key: str) -> None:
    rows = [
        {
            "index": n.index,
            "layer": n.layer,
            "kind": n.kind,
            "title": n.title or (n.text[:60] if n.text else ""),
            "tokens": n.n_tokens,
        }
        for n in sorted(tree.all_nodes.values(), key=lambda n: n.index)
    ]
    event = st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"{selected_key}__table",
    )
    selected_rows = event.selection.rows if event is not None else []
    if selected_rows:
        st.session_state[selected_key] = rows[selected_rows[0]]["index"]


def _render_node_detail(tree, node_id: int) -> None:
    node = tree.all_nodes[node_id]
    info = _describe_node(tree, node_id)
    st.markdown(f"**{' › '.join(node.heading_path) if node.heading_path else node.title or f'Node {node_id}'}**")
    st.badge(f"L{node.layer} · {node.kind} · {node.n_tokens} tok")
    if info:
        st.caption(
            f"Con: {info['n_children']} · Anh em: {info['n_siblings']} · "
            f"Có cha: {'có' if info['has_parent'] else 'không'}"
        )
    st.write(node.text)
    if tree.source_text:
        lines = tree.source_text.split("\n")
        excerpt = "\n".join(lines[node.body_line_start : node.body_line_end])
        with st.expander("Đoạn nguồn (đánh số dòng)"):
            st.code(
                "\n".join(
                    f"{i + node.body_line_start + 1:>5} | {line}"
                    for i, line in enumerate(excerpt.split("\n"))
                ),
                language=None,
            )
