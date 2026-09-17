"""pages/document.py — Chi tiết tài liệu: Tổng quan / Cây / Nguồn (xem
plan6.md D, D2, D3)."""

from __future__ import annotations

from uuid import UUID

import streamlit as st

from slidegen_rag.tree.stats import build_stats_from_dict, format_ingest_report
from ui import data as ui_data
from ui.components.tree_browser import render_tree_browser

document_id_str = ui_data.resolve_id_param("focus_document_id")
focus_node_id = st.session_state.pop("focus_node_id", None)

if not document_id_str:
    st.info("Chưa chọn tài liệu nào.")
    if st.button("← Về Projects"):
        st.switch_page("pages/projects.py")
    st.stop()

document_id = UUID(document_id_str)
document_repo = ui_data.get_document_repo()
doc = ui_data.run_async(document_repo.get_document(document_id))
if doc is None or doc.deleted_at is not None:
    st.error("Tài liệu không tồn tại hoặc đã bị xoá.")
    if st.button("← Về Projects"):
        st.switch_page("pages/projects.py")
    st.stop()

st.title(f"📄 D{doc.ref_no} · {doc.title}")
col_back, col_rename = st.columns([1, 5])
with col_back:
    if st.button("← Project"):
        st.session_state["active_project_id"] = str(doc.project_id)
        st.query_params["active_project_id"] = str(doc.project_id)
        st.switch_page("pages/project_home.py")

building = doc.status in ("queued", "parsing", "summarizing", "embedding")
tab_overview, tab_tree, tab_source = st.tabs(["Tổng quan", "Cây", "Nguồn"])

with tab_overview:
    st.write(f"**sha1**: `{doc.source_sha1}`")
    if doc.status == "failed":
        st.error("Lần build gần nhất thất bại.")
    if doc.outline:
        st.subheader("Mục lục")
        for title in doc.outline:
            st.markdown(f"- {title}")
    if doc.doc_summary:
        with st.expander("Tóm tắt tài liệu (node root)"):
            st.write(doc.doc_summary)

    st.subheader("Lịch sử build")
    indexes = ui_data.run_async(document_repo.list_indexes(document_id))
    for idx in indexes:
        badge = "✅" if idx.status == "ready" else ("❌" if idx.status == "failed" else "↩️")
        with st.expander(f"{badge} Lần {idx.attempt_no} · {idx.status} · {idx.created_at:%Y-%m-%d %H:%M}"):
            if idx.error_message:
                st.code(idx.error_message, language=None)
            if idx.build_stats:
                stats = build_stats_from_dict(idx.build_stats)
                st.code(format_ingest_report(stats), language=None)

    if doc.status == "failed" or not building:
        if st.button("🔁 Build lại"):
            src_path = ui_data.source_md_path(doc.project_id, doc.id)
            if not src_path.exists():
                st.error(f"Không tìm thấy file nguồn: {src_path}")
            else:
                markdown_text = src_path.read_text(encoding="utf-8")
                job = ui_data.start_ingest_job(
                    document_repo, project_id=doc.project_id, document=doc, markdown_text=markdown_text
                )
                st.session_state.setdefault("active_ingest_jobs", {})[str(doc.id)] = job.job_id
                st.rerun()

active_jobs = st.session_state.get("active_ingest_jobs", {})
job_id = active_jobs.get(str(doc.id))
if job_id is not None:
    from ui.jobs import get_job_registry

    job = get_job_registry().get(job_id)
    if job is not None:
        snap = job.snapshot()
        if snap["state"] in ("done", "failed", "cancelled"):
            ui_data.finalize_ingest_job(document_repo, job)
            active_jobs.pop(str(doc.id), None)
            st.session_state["active_ingest_jobs"] = active_jobs
            st.rerun()
        else:
            total = max(snap["total_units"], 1)
            st.progress(min(snap["done_units"] / total, 1.0), text=snap["current_label"] or "Đang xử lý…")
            building = True

with tab_tree:
    if building:
        st.info("Tài liệu đang được index — tab này sẽ sẵn sàng sau khi build xong.")
    elif doc.active_index_id is None:
        st.info("Chưa có bản index nào sẵn sàng.")
    else:
        index_row = ui_data.run_async(document_repo.get_active_index(document_id))
        if index_row is None or not index_row.artifact_path:
            st.error("Không tìm thấy artifact_path của bản index active.")
        else:
            import os

            mtime_ns = os.stat(index_row.artifact_path).st_mtime_ns
            tree = ui_data.load_tree_cached(index_row.artifact_path, mtime_ns)
            render_tree_browser(tree, key_prefix=f"tree_{document_id}", focus_node_id=focus_node_id)

with tab_source:
    if building:
        st.info("Tài liệu đang được index — tab này sẽ sẵn sàng sau khi build xong.")
    elif doc.active_index_id is None:
        st.info("Chưa có bản index nào sẵn sàng.")
    else:
        index_row = ui_data.run_async(document_repo.get_active_index(document_id))
        if index_row is None or not index_row.artifact_path:
            st.error("Không tìm thấy artifact_path của bản index active.")
        else:
            import os

            mtime_ns = os.stat(index_row.artifact_path).st_mtime_ns
            tree = ui_data.load_tree_cached(index_row.artifact_path, mtime_ns)
            lines = (tree.source_text or "").split("\n")
            numbered = "\n".join(f"{i + 1:>5} | {line}" for i, line in enumerate(lines))
            st.code(numbered, language=None)
