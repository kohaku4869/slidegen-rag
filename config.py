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
# RAPTOR Tree Builder Config (raptor_tree package — GMM + UMAP clustering)
# ─────────────────────────────────────────────────────────────────────────────

class RaptorTreeConfig:
    """
    Tham số cho quá trình xây dựng cây theo đúng thuật toán paper RAPTOR
    (Sarthi et al.): chunk text cố định + GMM clustering (soft) với UMAP giảm chiều,
    đệ quy tới khi còn 1 root. Không phụ thuộc cấu trúc heading như TreeConfig.
    """

    # ── Embedding (dùng chung tham số với TreeConfig để nhất quán) ───────────
    EMBEDDING_MODEL_NAME: str = TreeConfig.EMBEDDING_MODEL_NAME
    EMBEDDING_MODEL_KEY: str = "SBERT"
    CLUSTER_EMBEDDING_MODEL: str = "SBERT"

    # ── Chunking (leaf nodes) ──────────────────────────────────────────────
    # Số token tối đa mỗi chunk lá (theo paper, thường 100)
    MAX_TOKENS: int = 100
    CHUNK_OVERLAP: int = 0

    # ── Clustering (GMM + UMAP) ────────────────────────────────────────────
    NUM_LAYERS: int = 5
    # Số chiều giảm còn lại sau UMAP; cũng là ngưỡng "quá ít node để cluster"
    REDUCTION_DIMENSION: int = 10
    # Ngưỡng xác suất GMM để gán node vào 1 cluster (soft assignment)
    CLUSTER_THRESHOLD: float = 0.1
    # Giới hạn số cluster tối đa khi tìm kiếm BIC-optimal
    MAX_CLUSTERS: int = 50
    # Nếu tổng token 1 cluster vượt ngưỡng này, đệ quy tách nhỏ cluster trước khi tóm tắt
    MAX_LENGTH_IN_CLUSTER: int = 3500

    # ── Summarization ──────────────────────────────────────────────────────
    SUMMARIZATION_LENGTH: int = TreeConfig.SUMMARIZATION_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# Tree Retriever Config
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverConfig:
    """Tham số cho quá trình truy xuất từ cây (retrieval)."""

    # Cosine similarity threshold khi dùng mode "threshold"
    THRESHOLD: float = 0.5

    # Số node tối đa trả về trong một lần truy xuất. Giảm 10 -> 3: retrieve_node
    # giờ phân rã câu hỏi thành nhiều facet, mỗi facet 1 tool call hẹp — top_k
    # nhỏ giữ mỗi call gọn (trước đây bug ở tree_retriever.py ghi đè cứng
    # top_k=20 nên giá trị này chưa từng có tác dụng thật).
    TOP_K: int = 3

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
# Observability Config (LangSmith tracing + trajectory logging)
# ─────────────────────────────────────────────────────────────────────────────

def _env_flag(name: str, default: bool = False) -> bool:
    """Đọc biến môi trường dạng boolean ('true'/'1'/'yes'/'on')."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


class LangSmithConfig:
    """
    Tham số cho LangSmith tracing và ghi trajectory của Agent.

    Toàn bộ đọc từ .env. Nguyên tắc fail-soft: thiếu API key hoặc lỗi mạng
    LangSmith KHÔNG được làm hỏng lượt chạy Agent — chỉ log cảnh báo rồi tắt.
    """

    # Endpoint LangSmith (mặc định là cloud SaaS)
    ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()

    # API key; rỗng => tự động tắt tracing
    API_KEY: str = os.getenv("LANGSMITH_API_KEY", "").strip()

    # Tên project hiển thị trên LangSmith UI
    PROJECT: str = os.getenv("LANGSMITH_PROJECT", "slidegen-rag").strip()

    # Bật tracing: cần LANGSMITH_TRACING=true VÀ có API key
    ENABLED: bool = _env_flag("LANGSMITH_TRACING", default=False) and bool(API_KEY)

    # Gọi thêm 1 read API để lấy URL trace ghi vào trajectory.txt.
    # Tắt đi nếu muốn chạy nhanh hơn — run_id vẫn luôn được ghi và tra được trên UI.
    RESOLVE_RUN_URL: bool = _env_flag("LANGSMITH_RESOLVE_RUN_URL", default=True)

    # ── Trajectory logging (độc lập với LangSmith, luôn chạy được offline) ──
    TRAJECTORY_ENABLED: bool = _env_flag("TRAJECTORY_ENABLED", default=True)

    # Số ký tự output tool giữ lại làm preview trong trajectory
    TRAJECTORY_TOOL_OUTPUT_PREVIEW: int = 200

    # Tên file trajectory sinh ra trong data/test/<topic>/
    TRAJECTORY_FILENAME: str = "trajectory.txt"


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
