"""pages/project_home.py — Hai cột Nguồn + Phiên chat của một project (xem
plan6.md D)."""

from __future__ import annotations

import hashlib
import re
from uuid import UUID

import streamlit as st

from slidegen_rag.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    EmptyDocumentError,
    UnsupportedDocumentError,
    to_markdown,
)
from slidegen_rag.services.jobs import get_job_registry
from ui import data as ui_data
from ui.layout import breadcrumb

_STATUS_BADGE = {
    "queued": ("Trong hàng đợi", "gray"),
    "parsing": ("Đang phân tích", "blue"),
    "summarizing": ("Đang tóm tắt", "blue"),
    "embedding": ("Đang embed", "blue"),
    "ready": ("Sẵn sàng", "green"),
    "failed": ("Lỗi", "red"),
    "superseded": ("Bản cũ", "gray"),
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
    return slug or "tai-lieu"


project_id_str = ui_data.resolve_id_param("active_project_id")
if not project_id_str:
    st.info("Chưa chọn project nào.")
    if st.button("← Về danh sách project"):
        st.switch_page("pages/projects.py")
    st.stop()

project_id = UUID(project_id_str)
project_repo = ui_data.get_project_repo()
document_repo = ui_data.get_document_repo()

project = ui_data.run_async(project_repo.get_project(project_id))
if project is None or project.deleted_at is not None:
    st.error("Project không tồn tại hoặc đã bị xoá.")
    if st.button("← Về danh sách project"):
        st.switch_page("pages/projects.py")
    st.stop()

breadcrumb(("Projects", "pages/projects.py"), (project.name, None))


@st.cache_data(show_spinner="Đang đọc file…", max_entries=8)
def _convert_upload(name: str, data: bytes) -> tuple[str | None, str | None, str | None]:
    """Convert + sha1 một lần cho mỗi nội dung file. Dialog rerun ở MỌI thao tác
    (gõ tiêu đề, bấm nút) — không cache thì pypdf/python-docx chạy lại mỗi lần,
    nút Upload hiện rất trễ. Trả `(text, sha1, lỗi)`: lỗi cũng được cache."""
    try:
        text = to_markdown(name, data)
    except (UnsupportedDocumentError, EmptyDocumentError) as e:
        return None, None, str(e)
    return text, hashlib.sha1(text.strip().encode("utf-8")).hexdigest(), None


@st.dialog("Upload tài liệu")
def _upload_dialog() -> None:
    uploaded = st.file_uploader(
        "Chọn file", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS]
    )
    if uploaded is None:
        return
    text, source_sha1, error = _convert_upload(uploaded.name, uploaded.getvalue())
    if error is not None:
        st.warning(error)
        return
    title = st.text_input("Tiêu đề", value=uploaded.name.rsplit(".", 1)[0])
    existing = ui_data.run_async(
        document_repo.find_by_sha1(project_id=project_id, source_sha1=source_sha1)
    )
    if existing is not None:
        st.warning(f"Tài liệu đã có: D{existing.ref_no} · {existing.title}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mở tài liệu đã có", width="stretch"):
                st.session_state["focus_document_id"] = str(existing.id)
                st.switch_page("pages/document.py")
        with col2:
            if st.button("Build lại", width="stretch"):
                job = ui_data.start_ingest_job(
                    document_repo, project_id=project_id, document=existing, markdown_text=text
                )
                st.session_state.setdefault("active_ingest_jobs", {})[str(existing.id)] = job.job_id
                st.rerun()
        return
    if st.button("Upload", type="primary", disabled=not title.strip()):
        doc = ui_data.run_async(
            document_repo.create_document(
                project_id=project_id,
                doc_slug=_slugify(uploaded.name.rsplit(".", 1)[0]),
                title=title.strip(),
                source_sha1=source_sha1,
                source_bytes=len(text.encode("utf-8")),
            )
        )
        job = ui_data.start_ingest_job(
            document_repo, project_id=project_id, document=doc, markdown_text=text
        )
        st.session_state.setdefault("active_ingest_jobs", {})[str(doc.id)] = job.job_id
        ui_data.invalidate_catalog()
        st.rerun()


ready_docs = [d for d in ui_data.list_documents_cached(str(project_id)) if d.status == "ready"]

title_col, chat_col = st.columns([4, 1], vertical_alignment="center")
title_col.title(project.name)
if chat_col.button(
    "Chat mới",
    icon=":material/chat:",
    type="primary",
    width="stretch",
    disabled=not ready_docs,
    help=None if ready_docs else "Cần ít nhất 1 tài liệu sẵn sàng (status ready)",
):
    session_repo = ui_data.get_session_repo()
    state = ui_data.run_async(
        session_repo.create_session(project_id=project_id, user_id=ui_data.LOCAL_USER_ID)
    )
    ui_data.run_async(session_repo.set_session_scope(state.id, [d.id for d in ready_docs]))
    ui_data.invalidate_catalog()
    st.session_state["active_session_id"] = str(state.id)
    st.query_params["active_session_id"] = str(state.id)
    st.switch_page("pages/chat.py")

tab_docs, tab_sessions = st.tabs(["Tài liệu", "Phiên chat"])

with tab_docs:
    if st.button("Upload tài liệu", icon=":material/upload:"):
        _upload_dialog()

    documents = ui_data.list_documents_cached(str(project_id))
    active_jobs: dict[str, str] = st.session_state.get("active_ingest_jobs", {})

    if not documents:
        st.info("Chưa có tài liệu nào trong project này. Upload file .md/.docx/.pdf để bắt đầu.")
    else:

        @st.fragment(run_every=1.0 if active_jobs else None)
        def _render_documents() -> None:
            registry = get_job_registry()
            still_active = dict(active_jobs)
            docs = ui_data.list_documents_cached(str(project_id))
            for doc in docs:
                job_id = still_active.get(str(doc.id))
                with st.container(border=True):
                    info_col, status_col, action_col = st.columns(
                        [4, 3, 1.4], vertical_alignment="center"
                    )
                    info_col.markdown(f"**{doc.title}**")
                    meta = f"D{doc.ref_no}"
                    if doc.total_nodes:
                        meta += f" · {doc.total_nodes} node · {doc.num_layers} layer"
                    info_col.caption(meta)

                    if job_id is not None:
                        job = registry.get(job_id)
                        if job is not None:
                            snap = job.snapshot()
                            if snap["state"] in ("done", "failed", "cancelled"):
                                ui_data.finalize_ingest_job(document_repo, job)
                                still_active.pop(str(doc.id), None)
                                st.session_state["active_ingest_jobs"] = still_active
                                st.rerun()
                            else:
                                total = max(snap["total_units"], 1)
                                status_col.progress(
                                    min(snap["done_units"] / total, 1.0),
                                    text=snap["current_label"] or "Đang xử lý…",
                                )
                                if action_col.button("Huỷ", key=f"cancel_{doc.id}", width="stretch"):
                                    registry.request_cancel(job_id)
                                continue
                    label, color = _STATUS_BADGE.get(doc.status, (doc.status, "gray"))
                    status_col.badge(label, color=color)
                    if action_col.button("Chi tiết", key=f"open_doc_{doc.id}", width="stretch"):
                        st.session_state["focus_document_id"] = str(doc.id)
                        st.switch_page("pages/document.py")
                    if doc.status == "failed":
                        st.caption("Build lỗi — mở chi tiết để xem và build lại.")

        _render_documents()

with tab_sessions:
    sessions = ui_data.list_sessions_cached(str(project_id))
    if not sessions:
        st.info('Chưa có phiên chat nào. Bấm "Chat mới" để bắt đầu.')
    for s_ in sessions:
        with st.container(border=True):
            info_col, action_col = st.columns([5, 1.4], vertical_alignment="center")
            info_col.markdown(f"**{s_.title or f'Phiên {str(s_.id)[:8]}'}**")
            info_col.caption(f"Hoạt động gần nhất: {s_.last_active_at:%d/%m/%Y %H:%M}")
            if action_col.button("Mở", key=f"open_session_{s_.id}", width="stretch"):
                st.session_state["active_session_id"] = str(s_.id)
                st.query_params["active_session_id"] = str(s_.id)
                st.switch_page("pages/chat.py")
