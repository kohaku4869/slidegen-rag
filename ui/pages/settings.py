"""pages/settings.py — Trạng thái DB, revision alembic, model/endpoint,
đường dẫn data (xem plan6.md D). `app.py::ensure_database_configured()` đã
chặn app từ trước nếu `DATABASE_URL` rỗng/không kết nối được — trang này chỉ
hiển thị, không gate lại."""

from __future__ import annotations

import streamlit as st

from slidegen_rag.config import (
    DATA_INDEX_DIR,
    DATA_SRC_DIR,
    ROOT_DIR,
    LLMConfig,
    MemoryConfig,
    SummarizationConfig,
    TreeConfig,
)
from slidegen_rag.memory.engine import ping
from ui import data as ui_data

st.title("Settings")

st.subheader("Cơ sở dữ liệu")
try:
    ui_data.run_async(ping())
    st.success("Kết nối Postgres OK")
except Exception as e:
    st.error(f"Không kết nối được: {e}")

revision = ui_data.get_alembic_revision()
st.caption(f"Alembic revision hiện tại: `{revision or 'không rõ'}`")
st.caption(f"DATABASE_URL: `{'đã cấu hình' if MemoryConfig.DATABASE_URL else 'chưa cấu hình'}`")

st.subheader("Model")
st.caption(f"Embedding: `{TreeConfig.EMBEDDING_MODEL_NAME}` (key `{TreeConfig.EMBEDDING_MODEL_KEY}`)")
st.caption(f"Embedding endpoint: `{TreeConfig.EMBEDDING_BASE_URL or '(local/default)'}`")
st.caption(
    f"Summarization: `{SummarizationConfig.MODEL}` "
    f"(fallback `{SummarizationConfig.FALLBACK_MODEL}`)"
)
st.caption(f"LLM agent: `{LLMConfig.MODEL}`")

st.subheader("Đường dẫn dữ liệu")
st.caption(f"Root: `{ROOT_DIR}`")
st.caption(f"data/src (CLI): `{DATA_SRC_DIR}`")
st.caption(f"data/index (CLI): `{DATA_INDEX_DIR}`")
st.caption(f"data/projects (UI): `{ROOT_DIR / 'data' / 'projects'}`")
