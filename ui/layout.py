"""layout.py — Khung chung cho mọi trang: sidebar điều hướng cố định (chỉ
Projects/Settings) và breadcrumb. Các trang cần ngữ cảnh (project, chat, tài
liệu) KHÔNG hiện trong sidebar — chỉ vào được qua breadcrumb/nút trong trang,
vì chúng vô nghĩa nếu chưa chọn project/phiên."""

from __future__ import annotations

import streamlit as st


def render_sidebar_nav() -> None:
    with st.sidebar:
        st.markdown("### Slidegen RAG")
        st.page_link("pages/projects.py", label="Projects", icon=":material/folder:")
        st.page_link("pages/settings.py", label="Settings", icon=":material/settings:")
        st.divider()


def breadcrumb(*items: tuple[str, str | None]) -> None:
    """`items`: `(nhãn, đường dẫn page | None)`. Mục cuối (hiện tại) truyền
    `None`. Mục có page render thành nút link nhỏ, ngăn bởi dấu ›."""
    with st.container(horizontal=True, vertical_alignment="center", gap="xxsmall"):
        for label, page in items:
            if page is None:
                st.markdown(f"**{label}**")
            elif st.button(f"{label} ›", key=f"crumb_{label}_{page}", type="tertiary"):
                st.switch_page(page)
