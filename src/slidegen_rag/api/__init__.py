"""api — Lớp REST API (FastAPI) bọc quanh agent/tree/memory hiện có.

Đây là điểm vào thứ ba (bên cạnh CLI `main.py`/`ingest.py` và Streamlit
`ui/`) để một frontend khác (web/mobile) gắn vào qua HTTP thay vì gọi thẳng
code Python. Không sửa `ui/` hay `slidegen_rag.evaluate` — chỉ import những
hàm/class thuần (không phụ thuộc Streamlit) từ `ui/`.
"""
