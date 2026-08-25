"""
config.py — Central configuration for the slidegen-rag project.

Chỉnh sửa file này để thay đổi các tham số của:
  - Tree building  (TreeConfig)
  - Tree retrieval (RetrieverConfig)
  - Agent / LLM    (LLMConfig)
  - Summarization  (SummarizationConfig)
  - Đường dẫn dữ liệu

Các module khác import từ đây, không hardcode giá trị.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Thư mục gốc của project (nơi chứa config.py này)
ROOT_DIR = Path(__file__).parent

# Thư mục chứa các file Markdown nguồn cần index
DATA_SRC_DIR = ROOT_DIR / "data" / "src"

# Thư mục lưu tree index đã build (.pkl)
DATA_INDEX_DIR = ROOT_DIR / "data" / "index"

# ─────────────────────────────────────────────────────────────────────────────
# Tree Builder Config
# ─────────────────────────────────────────────────────────────────────────────

class TreeConfig:
    """
    Tham số cho quá trình xây dựng cây phân cấp (indexing).
    Cây và các node lá (Leaf nodes) được xây dựng tự động dựa trên cấu trúc phân cấp (Markdown headers).
    """

    # ── Embedding ──────────────────────────────────────────────────────────
    # Model SentenceTransformer dùng để tạo embedding cho các node
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # URL endpoint ngrok / server embedding từ xa (ưu tiên gọi trước, fallback về local nếu lỗi)
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "").strip()
    EMBEDDING_TIMEOUT: int = 60

    # Key định danh model trong hệ thống
    EMBEDDING_MODEL_KEY: str = "SBERT"
    CLUSTER_EMBEDDING_MODEL: str = "SBERT"

    # ── Summarization ──────────────────────────────────────────────────────
    # Độ dài tóm tắt tối đa (số token) khi LLM tóm tắt các node cha
    SUMMARIZATION_LENGTH: int = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Tree Retriever Config
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverConfig:
    """Tham số cho quá trình truy xuất từ cây (retrieval)."""

    # Cosine similarity threshold khi dùng mode "threshold"
    THRESHOLD: float = 0.5

    # Số node tối đa trả về trong một lần truy xuất
    TOP_K: int = 10

    # Chiến lược chọn node: "top_k" hoặc "threshold"
    SELECTION_MODE: str = "top_k"

    # Key embedding model dùng để tính similarity với query
    CONTEXT_EMBEDDING_MODEL: str = "SBERT"

    # Số token tối đa của context trả về từ retrieve()
    MAX_TOKENS: int = 4000

    # Số layer tối đa traverse khi collapse_tree=False (None = dùng tất cả)
    NUM_LAYERS: int = None

    # Layer bắt đầu tìm kiếm (None = bắt đầu từ layer cao nhất)
    START_LAYER: int = None


# ─────────────────────────────────────────────────────────────────────────────
# Agent / LLM Config
# ─────────────────────────────────────────────────────────────────────────────

class LLMConfig:
    """Tham số cho LLM sử dụng trong Agent (Google Gemini)."""

    # Model ID Gemini (ví dụ: "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.0")
    MODEL: str = "gemini-2.5-flash"

    # Nhiệt độ sampling (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE: float = 0.3

    # Nucleus sampling
    TOP_P: float = 0.95

    # Số token tối đa LLM sinh ra trong một lần gọi
    MAX_TOKENS: int = 16384

    # Timeout HTTP (giây)
    TIMEOUT: int = 180

    # Số lần retry khi gặp lỗi tạm thời
    MAX_RETRIES: int = 3

    # Temperature riêng cho bước retrieve (cần output tool_call ổn định hơn)
    RETRIEVE_TEMPERATURE: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Summarization Model Config
# ─────────────────────────────────────────────────────────────────────────────

class SummarizationConfig:
    """Tham số cho LLM dùng để tóm tắt node trong quá trình build tree."""

    # Model ID trên NVIDIA NIM / vLLM local
    MODEL: str = "deepseek-ai/deepseek-v4-flash"

    # Model Fallback khi gặp lỗi tràn context length (400 Bad Request / Max tokens)
    FALLBACK_MODEL: str = "gemini-2.5-flash"

    # Temperature khi tóm tắt (thường để thấp để output nhất quán)
    TEMPERATURE: float = 0.3
