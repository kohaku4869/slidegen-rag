"""pages/projects.py — Màn hình mặc định: lưới card project (xem plan6.md D)."""

from __future__ import annotations

import streamlit as st

from ui import data as ui_data

head_title, head_action = st.columns([4, 1], vertical_alignment="center")
head_title.title("Projects")


@st.dialog("Tạo project mới")
def _create_project_dialog() -> None:
    name = st.text_input("Tên project")
    if st.button("Tạo", type="primary", disabled=not name.strip()):
        try:
            ui_data.run_async(
                ui_data.get_project_repo().create_project(
                    owner_id=ui_data.LOCAL_USER_ID, name=name.strip()
                )
            )
        except Exception as e:
            st.error(f"Không tạo được project (tên có thể đã tồn tại): {e}")
            return
        ui_data.invalidate_catalog()
        st.rerun()


@st.dialog("Đổi tên project")
def _rename_project_dialog(project) -> None:
    name = st.text_input("Tên mới", value=project.name)
    if st.button("Lưu", type="primary", disabled=not name.strip()):
        ui_data.run_async(ui_data.get_project_repo().rename_project(project.id, name=name.strip()))
        ui_data.invalidate_catalog()
        st.rerun()


@st.dialog("Xoá project")
def _delete_project_dialog(project) -> None:
    st.warning(f"Xoá project **{project.name}**? Gõ đúng tên để xác nhận.")
    confirm = st.text_input("Tên project")
    if st.button("Xoá vĩnh viễn", type="primary", disabled=confirm != project.name):
        ui_data.run_async(ui_data.get_project_repo().soft_delete_project(project.id))
        ui_data.invalidate_catalog()
        st.rerun()


try:
    projects = ui_data.list_projects_cached(str(ui_data.LOCAL_USER_ID))
except Exception as e:
    st.error(f"Không đọc được danh sách project: {e}")
    if st.button("Thử lại"):
        ui_data.invalidate_catalog()
        st.rerun()
    st.stop()

if head_action.button("Tạo project", icon=":material/add:", type="primary", width="stretch"):
    _create_project_dialog()

if not projects:
    st.info('Chưa có project nào. Bấm "Tạo project" để bắt đầu upload tài liệu.')
    st.stop()

cols = st.columns(3)
for i, project in enumerate(projects):
    stats = ui_data.project_stats_cached(str(project.id))
    with cols[i % 3], st.container(border=True):
        st.markdown(f"#### {project.name}")
        meta = f"{stats.document_count} tài liệu · {stats.session_count} phiên chat"
        if stats.last_active_at:
            meta += f" · {stats.last_active_at:%d/%m/%Y %H:%M}"
        st.caption(meta)
        open_col, menu_col = st.columns([4, 1])
        if open_col.button("Mở", key=f"open_{project.id}", type="primary", width="stretch"):
            st.session_state["active_project_id"] = str(project.id)
            st.query_params["active_project_id"] = str(project.id)
            st.switch_page("pages/project_home.py")
        with menu_col.popover("", icon=":material/more_horiz:", width="stretch"):
            if st.button("Đổi tên", key=f"rename_{project.id}", type="tertiary", icon=":material/edit:"):
                _rename_project_dialog(project)
            if st.button("Xoá", key=f"delete_{project.id}", type="tertiary", icon=":material/delete:"):
                _delete_project_dialog(project)
