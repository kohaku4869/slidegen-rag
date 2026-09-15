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

# ── Paths ─────────────────────────────────────────────────────────────────

# Thư mục gốc của project (repo root — 3 cấp trên: config.py -> slidegen_rag -> src -> root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_SRC_DIR = ROOT_DIR / "data" / "src"

DATA_INDEX_DIR = ROOT_DIR / "data" / "index"

# ── Tree Builder Config ───────────────────────────────────────────────────

class TreeConfig:
    """
    Tham số cho quá trình xây dựng cây phân cấp (indexing).
    Cây và các node lá (Leaf nodes) được xây dựng tự động dựa trên cấu trúc phân cấp (Markdown headers).
    """

    # ── Embedding ──────────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # URL endpoint ngrok / server embedding từ xa (ưu tiên gọi trước, fallback về local nếu lỗi)
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "").strip()
    EMBEDDING_TIMEOUT: int = 60

    EMBEDDING_MODEL_KEY: str = "SBERT"
    CLUSTER_EMBEDDING_MODEL: str = "SBERT"

    # ── Summarization ──────────────────────────────────────────────────────
    # Độ dài tóm tắt tối đa (số token) khi LLM tóm tắt các node cha
    SUMMARIZATION_LENGTH: int = 2000


# ── Tree Retriever Config ─────────────────────────────────────────────────

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


# ── Agent / LLM Config ────────────────────────────────────────────────────

class LLMConfig:
    """Tham số cho LLM sử dụng trong Agent (Google Gemini)."""

    # Model ID Gemini (ví dụ: "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")
    # Đổi từ gemini-2.5-flash sang gemini-3.5-flash-lite: cùng bảng giá
    # $0.30/$2.50 per 1M token input/output (xem GEMINI_PRICING trong
    # agent/models/llm.py — không đổi cost/token), "flash-lite" nhắm tới
    # latency thấp hơn cho tác vụ high-throughput/agentic — CHƯA đo lại
    # benchmark pldc_v2 với model này, cần chạy lại để xác nhận latency/
    # chất lượng thực tế trước khi kết luận.
    MODEL: str = "gemini-3.5-flash-lite"

    # Nhiệt độ sampling (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE: float = 0.3

    # Nucleus sampling
    TOP_P: float = 0.95

    # Số token tối đa LLM sinh ra trong một lần gọi
    MAX_TOKENS: int = 16384

    # Trần RIÊNG cho generate_node, cố ý cao gấp đôi MAX_TOKENS ở trên — KHÔNG
    # phải nhầm lẫn. Đã đo được một câu 3-facet dùng hết out=16.084/16.384
    # (chạm trần) rồi bị cắt ngang giữa bảng so sánh, chỉ còn 1.327 ký tự nội
    # dung thật, điểm rơi 9.0 -> 2.75. Phần lớn ngân sách bị reasoning token
    # của Gemini ăn mất trước khi kịp sinh nội dung hiển thị, nên riêng bước
    # tổng hợp cuối phải nới; các node khác (output ngắn) giữ nguyên MAX_TOKENS.
    GENERATE_MAX_TOKENS: int = 32768

    # Timeout HTTP (giây)
    TIMEOUT: int = 180

    # Số lần retry khi gặp lỗi tạm thời
    MAX_RETRIES: int = 3

    # Temperature riêng cho bước retrieve (cần output tool_call ổn định hơn)
    RETRIEVE_TEMPERATURE: float = 0.0


# ── Agent Config (vòng lặp điều khiển của graph) ──────────────────────────


class AgentConfig:
    """Hằng số điều khiển luồng của graph.

    Chỉ đặt ở đây những hằng số có TỪ 2 NƠI TRỞ LÊN cùng đọc. Các knob chỉ một
    node dùng (ngưỡng grader, trần tool call, kích thước digest...) cố ý nằm
    lại ngay trong node đó, cạnh đoạn comment giải thích vì sao có giá trị đó —
    đẩy chúng ra file này chỉ làm khó hiểu hơn.
    """

    # Số lần retrieve tối đa trước khi bắt buộc generate dù context chưa đủ.
    # Hai nơi đọc: graph/router.py (quyết định dừng) và
    # observability/trajectory.py (hiển thị "n/MAX"). Trước đây trajectory
    # import ngược lên router — module quan sát phụ thuộc module điều khiển.
    MAX_RETRIEVE: int = 5


# ── Observability Config (LangSmith tracing + trajectory logging) ─────────


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


# ── Summarization Model Config ────────────────────────────────────────────

class SummarizationConfig:
    """Tham số cho LLM dùng để tóm tắt node trong quá trình build tree."""

    # Model ID trên NVIDIA NIM / vLLM local
    MODEL: str = "deepseek-ai/deepseek-v4-flash"

    # Model Fallback khi gặp lỗi tràn context length (400 Bad Request / Max tokens)
    FALLBACK_MODEL: str = "gemini-2.5-flash"

    # Temperature khi tóm tắt (thường để thấp để output nhất quán)
    TEMPERATURE: float = 0.3
