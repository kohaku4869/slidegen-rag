"""
Hằng số và thiết lập tiến trình dùng chung cho dashboard đánh giá.

Tách riêng khỏi logic để mọi module trong package cùng đọc một nguồn: đường
dẫn dữ liệu, ngưỡng màu hiển thị, giới hạn ký tự đưa vào LLM Judge, cùng các
thiết lập chỉ được phép chạy MỘT lần cho cả tiến trình (fix encoding console
Windows, matplotlib backend, logging, nạp .env).
"""

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ── Fix encoding console trên Windows ─────────────────────────────────────────
if sys.platform == "win32":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Đường dẫn & Cấu hình ─────────────────────────────────────────────────────
# repo root — 4 cấp trên: settings.py -> evaluate -> slidegen_rag -> src -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_TEST_DIR = ROOT_DIR / "data" / "test"
DATA_INDEX_DIR = ROOT_DIR / "data" / "index"

# Cache phụ trợ cho chế độ "chỉ chạy lại generate" — xem load_generate_state.
GENERATE_STATE_FILENAME = "generate_state.json"

# ── Layout thư mục run: data/test/<topic>/runs/<run_id>/... (xem plan5.md
# Phần C) — mỗi lần chạy tự chứa (self-contained) thay vì ghi đè answer.txt/
# chunks.txt/trajectory.txt phẳng trong data/test/<topic>/ như trước.
RUNS_DIRNAME = "runs"
QUESTION_FILENAME = "question.txt"
REFERENCE_FILENAME = "reference.txt"
ANSWER_FILENAME = "answer.txt"
CHUNKS_FILENAME = "chunks.txt"
SCORES_FILENAME = "scores.csv"
SUMMARY_FILENAME = "summary.json"
RUN_CONFIG_FILENAME = "run_config.json"
LATEST_POINTER_FILENAME = "LATEST.txt"
CONVERSATION_FILENAME = "conversation.json"
TURN_METRICS_FILENAME = "turn_metrics.json"
# %Y%m%d_%H%M%S sort lexicographically ĐÚNG theo thời gian — resolve_latest()
# dựa vào tính chất này khi LATEST.txt bị mất (xem runs.py).
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"

# Ngưỡng nhiễu khi so sánh 2 run (evaluate/compare.py) — dưới ngưỡng này coi
# là nhiễu đo lường, KHÔNG tô xanh/đỏ. Nhiễu judge đo được ±0.06 (xem README).
JUDGE_NOISE_DELTA = 0.1
COVERAGE_NOISE_DELTA = 1.0

load_dotenv(ROOT_DIR / ".env", override=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Matplotlib config ────────────────────────────────────────────────────────
matplotlib.use("Agg")
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]

# ── Ngưỡng màu hiển thị Dashboard (Thang điểm 10) ───────────────────────────
SCORE_GREEN = 8.5  # Xuất sắc
SCORE_AMBER = 7.0  # Đạt / Khá
COVERAGE_GREEN = 85.0
COVERAGE_AMBER = 70.0

# ── Giới hạn ký tự an toàn cho LLM Judge để tránh quá Context Window (8k) ────
# Primary judge có thể là model local với context nhỏ, nhưng evaluate_single_
# metric_with_judge tự động fallback sang Gemini (context lớn) nếu primary lỗi
# — nên nới ngưỡng vẫn an toàn, chỉ tăng khả năng fallback được gọi nhiều hơn
# một chút thay vì cắt oan câu trả lời dài nhưng hợp lệ.
_MAX_EVAL_CHUNKS_CHARS = 16_000
_MAX_EVAL_REF_CHARS = 8_000
_MAX_EVAL_ANS_CHARS = 12_000
