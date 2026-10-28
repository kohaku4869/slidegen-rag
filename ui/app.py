"""
app.py — Điểm vào Streamlit: `uv run streamlit run ui/app.py` (xem plan6.md
C1/D — `ui/` sống ở repo root, tách khỏi package backend `slidegen_rag`, xem
plan7.md). `st.navigation` + `st.Page` (multipage) thay vì sidebar
routing thủ công — mỗi màn hình một file, có URL thật và nút Back.

**KHÔNG import `slidegen_rag.evaluate.*`** ở bất kỳ đâu trong `ui/` — dashboard
Gradio đánh giá chạy tiến trình riêng (`find_available_port(7860)`), import
chéo sẽ kéo theo `LLMModel.reset_usage()`/`asyncio.run` xung đột với
`AgentRuntime` (xem A1). **Không import `slidegen_rag.main`** — nó chạy
`sys.stdout.reconfigure` + định nghĩa `argparse` CLI ở module scope; `ui/data.py`
tự dựng lại `RetrievalAugmentationConfig` (giống `main.py::load_ra`) thay vì
import lại từ đó.
"""

from __future__ import annotations

import logging
import sys

import streamlit as st


def configure_ui_logging() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


configure_ui_logging()

st.set_page_config(page_title="Slidegen RAG", page_icon=":material/menu_book:", layout="wide")

from ui.data import ensure_database_configured, warm_up_heavy_imports
from ui.layout import render_sidebar_nav

ensure_database_configured()
warm_up_heavy_imports()

pages = [
    st.Page("pages/projects.py", title="Projects", icon=":material/folder:", url_path="projects", default=True),
    st.Page("pages/project_home.py", title="Project", icon=":material/folder_open:", url_path="project"),
    st.Page("pages/chat.py", title="Chat", icon=":material/chat:", url_path="chat"),
    st.Page("pages/document.py", title="Tài liệu", icon=":material/description:", url_path="document"),
    st.Page("pages/settings.py", title="Settings", icon=":material/settings:", url_path="settings"),
]

# Nav ẩn: chỉ Projects/Settings hiện ở sidebar (xem `ui/layout.py`); các trang
# cần ngữ cảnh đi qua breadcrumb/nút trong trang.
nav = st.navigation(pages, position="hidden")
render_sidebar_nav()
nav.run()
