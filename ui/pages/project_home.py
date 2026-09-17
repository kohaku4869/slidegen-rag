"""pages/project_home.py — Hai cột Nguồn + Phiên chat của một project (xem
plan6.md D)."""

from __future__ import annotations

import hashlib
import re
from uuid import UUID

import streamlit as st

from ui import data as ui_data

_STATUS_BADGE = {
    "queued": ("⏳ Trong hàng đợi", "gray"),
    "parsing": ("⚙️ Đang phân tích", "blue"),
    "summarizing": ("⚙️ Đang tóm tắt", "blue"),
    "embedding": ("⚙️ Đang embed", "blue"),
    "ready": ("✅ Sẵn sàng", "green"),
    "failed": ("❌ Lỗi", "red"),
    "superseded": ("↩️ Bản cũ", "gray"),
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

st.title(f"🗂️ {project.name}")
if st.button("← Tất cả project"):
    st.switch_page("pages/projects.py")


@st.dialog("Upload tài liệu")
def _upload_dialog() -> None:
    uploaded = st.file_uploader("Chọn file .md", type=["md"])
    if uploaded is None:
        return
    text = uploaded.getvalue().decode("utf-8", errors="ignore")
    if not text.strip():
        st.warning("File rỗng.")
        return
    title = st.text_input("Tiêu đề", value=uploaded.name.rsplit(".", 1)[0])
    source_sha1 = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
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


col_docs, col_sessions = st.columns([3, 2])

with col_docs:
    st.subheader("Nguồn")
    if st.button("⬆️ Upload tài liệu"):
        _upload_dialog()

    documents = ui_data.list_documents_cached(str(project_id))
    active_jobs: dict[str, str] = st.session_state.get("active_ingest_jobs", {})

    if not documents:
        st.info("Chưa có tài liệu nào trong project này.")
    else:

        @st.fragment(run_every=1.0 if active_jobs else None)
        def _render_documents() -> None:
            from ui.jobs import get_job_registry

            registry = get_job_registry()
            still_active = dict(active_jobs)
            docs = ui_data.list_documents_cached(str(project_id))
            for doc in docs:
                job_id = still_active.get(str(doc.id))
                with st.container(border=True):
                    header_cols = st.columns([0.5, 3, 2, 1.5])
                    header_cols[0].write(f"D{doc.ref_no}")
                    if header_cols[1].button(doc.title, key=f"open_doc_{doc.id}"):
                        st.session_state["focus_document_id"] = str(doc.id)
                        st.switch_page("pages/document.py")

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
                                header_cols[2].progress(
                                    min(snap["done_units"] / total, 1.0),
                                    text=snap["current_label"] or "Đang xử lý…",
                                )
                                if header_cols[3].button("Huỷ", key=f"cancel_{doc.id}"):
                                    registry.request_cancel(job_id)
                                continue
                    label, color = _STATUS_BADGE.get(doc.status, (doc.status, "gray"))
                    header_cols[2].badge(label, color=color)
                    if doc.total_nodes:
                        header_cols[3].caption(f"{doc.total_nodes} node · {doc.num_layers} layer")
                    if doc.status == "failed":
                        st.caption("Build lỗi — mở tài liệu để xem chi tiết và build lại.")

        _render_documents()

with col_sessions:
    st.subheader("Phiên chat")
    ready_docs = [d for d in ui_data.list_documents_cached(str(project_id)) if d.status == "ready"]
    if not ready_docs:
        st.button("💬 Chat mới", disabled=True, help="Cần ít nhất 1 tài liệu sẵn sàng (status ready)")
    elif st.button("💬 Chat mới", type="primary"):
        session_repo = ui_data.get_session_repo()
        state = ui_data.run_async(
            session_repo.create_session(project_id=project_id, user_id=ui_data.LOCAL_USER_ID)
        )
        ui_data.run_async(session_repo.set_session_scope(state.id, [d.id for d in ready_docs]))
        ui_data.invalidate_catalog()
        st.session_state["active_session_id"] = str(state.id)
        st.query_params["active_session_id"] = str(state.id)
        st.switch_page("pages/chat.py")

    sessions = ui_data.list_sessions_cached(str(project_id))
    for s in sessions:
        with st.container(border=True):
            label = s.title or f"Phiên {str(s.id)[:8]}"
            if st.button(label, key=f"open_session_{s.id}", width="stretch"):
                st.session_state["active_session_id"] = str(s.id)
                st.query_params["active_session_id"] = str(s.id)
                st.switch_page("pages/chat.py")
            st.caption(f"Hoạt động gần nhất: {s.last_active_at:%Y-%m-%d %H:%M}")
