"""
components/citations.py — Hàng chip trích dẫn dưới mỗi câu trả lời (xem
plan6.md D1). `extract_citations` nằm ở `slidegen_rag.services.citations` (thuần, không
Streamlit); `render_*` vẽ UI và cần `merged_tree`/`document_repo` của phiên.
"""

from __future__ import annotations

import streamlit as st

from slidegen_rag.services.citations import extract_citations  # noqa: F401 — re-export
from slidegen_rag.tree.merge import resolve_node
from ui import data as ui_data


def render_live_citations(citations: list[dict], merged_tree, document_repo, *, key_prefix: str) -> None:
    """Trích dẫn của lượt VỪA chạy — node chắc chắn có trong `merged_tree`
    hiện tại (không cần kiểm staleness, khác `render_replayed_citations`)."""
    if not citations or merged_tree is None:
        return
    cols = st.columns(len(citations))
    for i, (col, c) in enumerate(zip(cols, citations)):
        node = merged_tree.all_nodes.get(c["node_id"])
        if node is None:
            continue
        resolved = resolve_node(merged_tree, c["node_id"])
        with col:
            _render_chip(node, resolved, merged_tree, document_repo, key=f"{key_prefix}_{i}_{c['node_id']}")


def render_replayed_citations(refs, merged_tree, document_repo, *, key_prefix: str) -> None:
    """Trích dẫn nạp lại từ Postgres (`message_citations`) — chỉ resolve chip
    nào `source_sha1` khớp bản index ĐANG active cho đúng tài liệu; còn lại
    render xám kèm tooltip "index đã build lại" (xem plan6.md D1, đúng logic
    `ConversationManager.cited_node_ids_last_turn`)."""
    if not refs:
        return
    cols = st.columns(len(refs))
    for i, (col, ref) in enumerate(zip(cols, refs)):
        with col:
            node = merged_tree.all_nodes.get(ref.node_index) if merged_tree is not None else None
            active_sha1 = None
            if ref.document_id is not None:
                idx = ui_data.run_async(document_repo.get_active_index(ref.document_id))
                active_sha1 = idx.source_sha1 if idx else None
            if node is None or not active_sha1 or active_sha1 != ref.source_sha1:
                st.badge(
                    f"{ref.doc_id or '?'} · index đã build lại", color="gray",
                    help="Tài liệu đã được index lại từ lượt hỏi này — node cũ không còn khớp.",
                )
                continue
            resolved = resolve_node(merged_tree, ref.node_index)
            _render_chip(node, resolved, merged_tree, document_repo, key=f"{key_prefix}_{i}_{ref.node_index}")


def _render_chip(node, resolved, merged_tree, document_repo, *, key: str) -> None:
    label = node.title or (node.heading_path[-1] if node.heading_path else f"Node {node.index}")
    with st.popover(label[:28] or "(không tiêu đề)"):
        if node.heading_path:
            st.caption(" › ".join(node.heading_path))
        st.badge(f"L{node.layer} · {node.kind} · {node.n_tokens} tok")
        st.write(node.text)

        col1, col2 = st.columns(2)
        with col1:
            if resolved is not None and st.button("Xem trong cây", key=f"{key}_tree"):
                doc_slice, orig_idx = resolved
                st.session_state["focus_document_id"] = doc_slice.doc_key
                st.session_state["focus_node_id"] = orig_idx
                st.switch_page("pages/document.py")
        with col2:
            show_key = f"{key}_show_source"
            if st.button("Xem dòng nguồn", key=f"{key}_src"):
                st.session_state[show_key] = not st.session_state.get(show_key, False)
            if st.session_state.get(show_key):
                excerpt = ui_data.get_source_excerpt(document_repo, merged_tree, node.index)
                st.code(excerpt or "(không tìm thấy nguồn)", language=None)
