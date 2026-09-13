"""
evaluate.py — Giao diện Dashboard đánh giá chất lượng hệ thống RAG / CRAG Agent.
Kiến trúc LLM-as-a-Judge trực tiếp với Pydantic Structured Output & Caching answer/chunks.

Các tiêu chuẩn đánh giá (Thang điểm 10):
  1. Accuracy (Độ chính xác thực tế so với Reference Answer)
  2. Completeness (Độ đầy đủ thông tin so với Reference Answer)
  3. Relevance (Độ tập trung, liên quan trực diện đến câu hỏi)
  4. Faithfulness (Tính trung thực / Đo lường ảo giác Hallucination dựa trên Context Chunks)

Mỗi tiêu chí được chấm bằng 1 PROMPT ĐỘC LẬP riêng (xem CRITERIA,
evaluate_single_metric_with_judge), thay vì gộp cả 4 tiêu chí vào 1 prompt —
tránh "halo effect" (điểm các tiêu chí kéo theo nhau khi LLM chấm cùng lúc).
4 lượt gọi này chạy song song (asyncio.gather) nên không tăng thời gian chấm
mỗi câu, chỉ tăng số lượt gọi LLM.

Tự động cache câu trả lời vào answer.txt và các chunks tài liệu sử dụng vào chunks.txt
để tái sử dụng nhanh chóng mà không cần chạy lại Agent mỗi lần đánh giá.

Xử lý tự động lỗi context length / timeout, cơ chế rút gọn context an toàn,
và loại bỏ câu đánh giá thất bại khỏi điểm tổng để không kéo tụt trung bình.

Sử dụng:
    python evaluate/evaluate.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── Fix encoding console trên Windows ─────────────────────────────────────────
if sys.platform == "win32":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Đường dẫn & Cấu hình ─────────────────────────────────────────────────────
# repo root — 4 cấp trên: evaluate.py -> evaluate -> slidegen_rag -> src -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_TEST_DIR = ROOT_DIR / "data" / "test"
DATA_INDEX_DIR = ROOT_DIR / "data" / "index"

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

# Một số câu trả lời hiếm khi rơi vào trạng thái lặp ký tự/cụm từ vô hạn (LLM
# degenerate output) — hàng chục nghìn ký tự lặp lại làm judge hiểu nhầm là
# câu trả lời dài nhưng hợp lệ rồi trừ điểm oan vì bị cắt giữa chừng. Rút gọn
# các đoạn lặp bất thường (>= 100 ký tự lặp liên tiếp) TRƯỚC khi áp dụng cắt
# theo độ dài, để giới hạn ký tự phía trên không bị "ăn" hết vào rác lặp.
_REPEAT_RUN_RE = re.compile(r"(.)\1{99,}", re.DOTALL)


def _collapse_pathological_repetition(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        run = match.group(0)
        return f"{match.group(1)}... [lặp {len(run)} lần liên tiếp, đã rút gọn] "

    return _REPEAT_RUN_RE.sub(_replace, text)


# =============================================================================
# 1. DATA MODELS & SCHEMA
# =============================================================================


class AnswerEval(BaseModel):
    """
    Đánh giá chất lượng câu trả lời bằng LLM-as-a-judge trên thang điểm 10.

    Được tổng hợp từ 4 lượt chấm ĐỘC LẬP (mỗi tiêu chí một prompt riêng, xem
    CRITERIA/evaluate_single_metric_with_judge) — không phải từ 1 prompt chấm
    cả 4 tiêu chí cùng lúc. Lý do: khi một prompt duy nhất yêu cầu LLM chấm
    nhiều tiêu chí cùng lúc, điểm các tiêu chí có xu hướng kéo theo nhau
    (halo effect) — ví dụ câu trả lời "nghe hay" dễ được chấm Faithfulness
    cao lây từ ấn tượng Relevance, dù hai tiêu chí độc lập về bản chất.
    Tách prompt giúp mỗi điểm phản ánh đúng riêng tiêu chí của nó.
    """

    accuracy: float = Field(
        description="Độ chính xác thực tế so với đáp án mẫu (0.0 đến 10.0). Sai thông tin cốt lõi bắt buộc điểm <= 3.0."
    )
    completeness: float = Field(
        description="Độ đầy đủ các ý so với đáp án mẫu (0.0 đến 10.0). Bao quát toàn bộ thông tin mới được 10.0."
    )
    relevance: float = Field(
        description="Độ liên quan và tập trung vào câu hỏi (0.0 đến 10.0). Trả lời trực diện, không lan man."
    )
    faithfulness: float = Field(
        description="Tính trung thực / Chống ảo giác (0.0 đến 10.0). Đo lường mức độ câu trả lời dựa trên context chunks được cung cấp, không bịa đặt (hallucination)."
    )
    feedback: str = Field(
        description="Nhận xét chi tiết bằng tiếng Việt: ghép từ feedback riêng của từng tiêu chí (accuracy/completeness/relevance/faithfulness)."
    )


class MetricEval(BaseModel):
    """Kết quả chấm MỘT tiêu chí duy nhất từ LLM Judge (xem CRITERIA)."""

    score: float = Field(
        description="Điểm 0.0 đến 10.0 cho đúng tiêu chí được giao — không xét tiêu chí khác."
    )
    feedback: str = Field(
        description="Nhận xét ngắn gọn (dưới 80 từ) bằng tiếng Việt, chỉ riêng cho tiêu chí này."
    )


# =============================================================================
# 2. HÀM ĐỌC & LƯU TRỮ DỮ LIỆU TEST (QUESTION, REFERENCE, ANSWER, CHUNKS)
# =============================================================================


def get_available_topics() -> list[str]:
    """Tìm tất cả thư mục chủ đề trong data/test/."""
    if not DATA_TEST_DIR.exists():
        return []
    topics = []
    for d in sorted(DATA_TEST_DIR.iterdir()):
        if (
            d.is_dir()
            and (d / "question.txt").exists()
            and (d / "reference.txt").exists()
        ):
            topics.append(d.name)
    return topics


def load_questions(file_path: Path) -> list[dict]:
    """Đọc danh sách câu hỏi từ question.txt."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.split(r"\n(?=\d+\.\s+)", content.strip())
    questions = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^(\d+)\.\s*(.+)$", block, re.DOTALL)
        if match:
            q_num = int(match.group(1))
            questions.append(
                {
                    "num": q_num,
                    "text": match.group(2).strip(),
                }
            )
    return questions


def load_references(file_path: Path) -> dict[int, str]:
    """Đọc đáp án tham chiếu từ reference.txt (-N. format)."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?:^|\n)-(\d+)\.\s*", content.strip())
    references = {}
    i = 1
    while i < len(blocks) - 1:
        num = int(blocks[i])
        text = blocks[i + 1].strip()
        references[num] = text
        i += 2
    return references


def load_answers(file_path: Path) -> dict[int, str]:
    """Đọc câu trả lời đã lưu từ answer.txt."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?:^|\n)-(\d+)\.\s*", content.strip())
    answers = {}
    if len(blocks) > 1:
        i = 1
        while i < len(blocks) - 1:
            num = int(blocks[i])
            text = blocks[i + 1].strip()
            answers[num] = text
            i += 2
    else:
        blocks = re.split(r"(?:^|\n)(\d+)\.\s*", content.strip())
        i = 1
        while i < len(blocks) - 1:
            num = int(blocks[i])
            text = blocks[i + 1].strip()
            answers[num] = text
            i += 2
    return answers


def save_answers(file_path: Path, answers: dict[int, str]):
    """Ghi toàn bộ câu trả lời ra answer.txt theo format -N."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for num in sorted(answers.keys()):
            text = answers[num].strip()
            f.write(f"-{num}. {text}\n\n")


def load_chunks(file_path: Path) -> dict[int, list[str]]:
    """Đọc danh sách các context chunks đã sử dụng từ chunks.txt."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    q_blocks = re.split(r"(?:^|\n)===\s*(?:CÂU HỎI|QUESTION)\s*(\d+)\s*===", content)
    chunks_dict = {}
    i = 1
    while i < len(q_blocks) - 1:
        q_num = int(q_blocks[i])
        raw_text = q_blocks[i + 1].strip()
        chunk_items = re.split(r"\[CHUNK\s*\d+\]\n?", raw_text)
        cleaned_chunks = [c.strip() for c in chunk_items if c.strip()]
        chunks_dict[q_num] = cleaned_chunks
        i += 2
    return chunks_dict


def save_chunks(file_path: Path, chunks: dict[int, list[str]]):
    """Ghi các context chunks đã sử dụng ra chunks.txt."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for num in sorted(chunks.keys()):
            f.write(f"=== CÂU HỎI {num} ===\n\n")
            doc_list = chunks[num]
            if not doc_list:
                f.write("[CHUNK 1]\n(Không có chunk nào được sử dụng)\n\n")
            else:
                for idx, doc in enumerate(doc_list, 1):
                    f.write(f"[CHUNK {idx}]\n{doc.strip()}\n\n")


def load_trajectory_texts(file_path: Path) -> dict[int, str]:
    """
    Đọc lại nội dung trajectory.txt (text thô, đã render sẵn) theo từng câu
    hỏi — dùng để hiển thị lại trên dashboard khi chạy từ cache (không cần
    parse thành Trajectory object, chỉ cần hiển thị).
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    q_blocks = re.split(r"(?:^|\n)===\s*(?:CÂU HỎI|QUESTION)\s*(\d+)\s*===\n*", content)
    texts: dict[int, str] = {}
    i = 1
    while i < len(q_blocks) - 1:
        q_num = int(q_blocks[i])
        texts[q_num] = q_blocks[i + 1].strip()
        i += 2
    return texts


# =============================================================================
# 3. KHỞI TẠO LLM EVALUATOR (JUDGE)
# =============================================================================


def create_evaluator_llm():
    """Khởi tạo LLM Judge (Primary: OpenAI compatible, Fallback: Gemini)."""
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    llm_primary = None
    llm_fallback = None

    if base_url and api_key:
        try:
            from langchain_openai import ChatOpenAI

            llm_primary = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                temperature=0,
                max_tokens=4096,
                timeout=120,
            )
            logger.info(f"[Judge] Primary LLM: OpenAI-compatible @ {base_url}")
        except Exception as e:
            logger.warning(f"[Judge] Không thể tạo primary LLM: {e}")

    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm_fallback = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_key,
                temperature=0,
                max_output_tokens=8192,
                timeout=120,
            )
            logger.info("[Judge] Fallback LLM: Gemini 2.5 Flash")
        except Exception as e:
            logger.warning(f"[Judge] Không thể tạo fallback Gemini LLM: {e}")

    return llm_primary, llm_fallback


# =============================================================================
# 4. CHẠY AGENT HOẶC ĐỌC TỪ CACHE
# =============================================================================


def get_answers_and_chunks_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
) -> tuple[list[dict], bool]:
    """
    Lấy danh sách câu trả lời và chunks cho chủ đề.
    Nếu file answer.txt và chunks.txt đã tồn tại đầy đủ và force_rerun=False,
    đọc trực tiếp từ file (cache).

    Returns:
        (results, was_cached)
    """
    from slidegen_rag.main import load_ra
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.tools.tools import (
        set_retrieval_augmentation,
        set_web_search_enabled,
    )
    from slidegen_rag.agent.observability.trajectory import (
        TrajectoryRecorder,
        Trajectory,
        make_initial_state,
        save_trajectories,
    )
    from slidegen_rag.config import LangSmithConfig

    question_file = DATA_TEST_DIR / topic / "question.txt"
    reference_file = DATA_TEST_DIR / topic / "reference.txt"
    answer_file = DATA_TEST_DIR / topic / "answer.txt"
    chunks_file = DATA_TEST_DIR / topic / "chunks.txt"
    trajectory_file = DATA_TEST_DIR / topic / LangSmithConfig.TRAJECTORY_FILENAME
    v2_path = DATA_INDEX_DIR / f"{topic}_v2.pkl"
    index_path = v2_path if v2_path.exists() else DATA_INDEX_DIR / f"{topic}.pkl"

    questions = load_questions(question_file)
    references = load_references(reference_file)

    if not questions:
        raise ValueError(f"Không tìm thấy câu hỏi nào trong {question_file}")

    cached_answers = load_answers(answer_file)
    cached_chunks = load_chunks(chunks_file)

    all_cached = (
        not force_rerun
        and len(cached_answers) >= len(questions)
        and all(q["num"] in cached_answers for q in questions)
    )

    if all_cached:
        if progress_callback:
            progress_callback(
                f"[{topic}] Đã nạp {len(questions)} câu từ cache (answer.txt & chunks.txt)"
            )
        logger.info(f"[{topic}] Sử dụng kết quả có sẵn từ answer.txt & chunks.txt")
        cached_trajectories = load_trajectory_texts(trajectory_file)
        results = []
        for q in questions:
            num = q["num"]
            results.append(
                {
                    "num": num,
                    "question": q["text"],
                    "response": cached_answers.get(num, ""),
                    "reference": references.get(num, ""),
                    "used_chunks": cached_chunks.get(num, []),
                    "trajectory": cached_trajectories.get(num),
                }
            )
        return results, True

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy index: {index_path}")

    if progress_callback:
        progress_callback(f"[{topic}] Đang khởi tạo CRAG Agent...")

    ra = load_ra(index_path)
    set_retrieval_augmentation(ra)
    # Benchmark closed-book: reference.txt được viết trên giả định agent CHỈ
    # dùng data/src/pldc.md. Nếu web_search được bind trong lúc eval, agent có
    # thể trả lời đúng-trong-thực-tế nhưng mâu thuẫn với reference (vd. tra ra
    # thời hiệu khởi kiện dân sự thật trong khi reference nói "tài liệu không
    # đề cập"), làm sai lệch điểm accuracy/faithfulness. Chạy thật (main.py)
    # vẫn giữ mặc định bật.
    set_web_search_enabled(False)
    graph = build_graph()
    recorder = TrajectoryRecorder(
        graph, tags=["evaluate", topic], metadata={"topic": topic}
    )

    results = []
    new_answers = {}
    new_chunks = {}
    new_trajectories: dict[int, Trajectory] = {}
    total = len(questions)

    for idx, q in enumerate(questions):
        num = q["num"]
        q_text = q["text"]
        ref = references.get(num, "")

        if progress_callback:
            progress_callback(
                f"[{topic}] Đang chạy Agent sinh câu trả lời {num}/{total}..."
            )

        logger.info(f"[{topic}] Agent đang trả lời câu {num}/{total}: {q_text[:70]}...")

        try:
            state_res, traj = recorder.run(
                make_initial_state(q_text),
                run_name=f"{topic}:Q{num}",
                metadata={
                    "topic": topic,
                    "question_num": num,
                },
            )
            answer = (state_res.get("generation") or "").strip()
            used_docs = state_res.get("documents", [])
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi chạy Agent câu {num}: {e}")
            answer = f"[LỖI AGENT] {e}"
            used_docs = []
            traj = Trajectory(query=q_text, started_at=datetime.now(), error=str(e))

        new_answers[num] = answer
        new_chunks[num] = used_docs
        new_trajectories[num] = traj

        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": ref,
                "used_chunks": used_docs,
                "trajectory": traj.render_text(),
            }
        )

    save_answers(answer_file, new_answers)
    save_chunks(chunks_file, new_chunks)
    save_trajectories(trajectory_file, new_trajectories, topic)
    logger.info(
        f"[{topic}] Đã lưu câu trả lời vào {answer_file}, chunks vào {chunks_file}, "
        f"trajectory vào {trajectory_file}"
    )

    return results, False


# =============================================================================
# 5. ĐÁNH GIÁ VỚI LLM-AS-A-JUDGE (THANG ĐIỂM 10)
# =============================================================================

# Preamble dùng chung cho cả 4 prompt — KHÔNG liệt kê rubric của các tiêu chí
# khác, để tránh LLM bị ảnh hưởng chéo giữa các tiêu chí (halo effect) khi
# chấm. Mỗi lượt gọi chỉ nhìn thấy đúng 1 rubric trong CRITERIA bên dưới.
JUDGE_PREAMBLE = """Bạn là một chuyên gia thẩm định và đánh giá chất lượng (LLM Judge) độc lập, khắt khe và công tâm cho hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ của bạn là đánh giá câu trả lời của hệ thống (Generated Answer) dựa trên câu hỏi (Question), đáp án mẫu chuẩn (Reference Answer) và các đoạn ngữ cảnh tài liệu đã được cung cấp (Retrieved Chunks).

QUAN TRỌNG: Bạn CHỈ chấm điểm cho DUY NHẤT MỘT tiêu chí được giao dưới đây trên thang điểm 10 (0.0 đến 10.0). KHÔNG xét đến các khía cạnh khác (ví dụ nếu đang chấm Accuracy thì không được vì câu trả lời "lan man" hay "có vẻ bịa đặt" mà hạ điểm — những khía cạnh đó thuộc tiêu chí khác, được chấm ở lượt riêng)."""

# Mỗi tiêu chí: key (khớp field AnswerEval), title (tên đầy đủ dùng trong
# prompt), label (tên ngắn dùng làm tiêu đề feedback ghép), rubric (mô tả
# thang điểm — nội dung y hệt JUDGE_SYSTEM_PROMPT trước khi tách prompt).
CRITERIA: list[dict] = [
    {
        "key": "accuracy",
        "label": "Accuracy",
        "title": "Accuracy (Độ chính xác thực tế)",
        "rubric": """Accuracy (Độ chính xác thực tế) [0.0 - 10.0]:
   - So sánh với Reference Answer.
   - 10.0: Hoàn hảo, tuyệt đối chính xác từng chi tiết so với đáp án mẫu.
   - 7.0 - 9.0: Đúng phần lớn ý chính, chỉ sai sót hoặc mơ hồ vài chi tiết nhỏ không đáng kể.
   - 4.0 - 6.0: Đúng một phần nhưng có thông tin sai lệch hoặc nhầm lẫn khái niệm cơ bản.
   - 0.0 - 3.0: Sai lệch nghiêm trọng về sự thật, hiểu sai câu hỏi hoặc trả lời hoàn toàn sai. (Bất kỳ câu nào sai sự thật cốt lõi bắt buộc điểm <= 3.0).""",
    },
    {
        "key": "completeness",
        "label": "Completeness",
        "title": "Completeness (Độ đầy đủ thông tin)",
        "rubric": """Completeness (Độ đầy đủ thông tin) [0.0 - 10.0]:
   - Mức độ bao quát tất cả các ý chính, căn cứ pháp lý và chi tiết quan trọng từ Reference Answer.
   - 10.0: Đầy đủ 100% tất cả các ý, phân nhánh hoặc điều kiện được nêu trong đáp án mẫu.
   - 7.0 - 9.0: Đạt hầu hết các ý quan trọng, chỉ thiếu một vài ý phụ hoặc ví dụ mở rộng.
   - 4.0 - 6.0: Chỉ nêu được khoảng một nửa số ý cần thiết.
   - 0.0 - 3.0: Thiếu nghiêm trọng, chỉ trả lời sơ sài hoặc bỏ sót phần lớn nội dung bắt buộc.""",
    },
    {
        "key": "relevance",
        "label": "Relevance",
        "title": "Relevance (Độ liên quan & Trọng tâm)",
        "rubric": """Relevance (Độ liên quan & Trọng tâm) [0.0 - 10.0]:
   - Mức độ tập trung vào đúng trọng tâm câu hỏi, không lan man, không chứa thông tin thừa ngoài phạm vi hỏi.
   - 10.0: Trực diện, mạch lạc, súc tích, không có câu chữ thừa.
   - 7.0 - 9.0: Đúng trọng tâm, có một ít thông tin bổ trợ nhưng vẫn liên quan mật thiết.
   - 4.0 - 6.0: Bắt đầu lan man, đưa ra nhiều thông tin không được hỏi.
   - 0.0 - 3.0: Lạc đề hoặc trả lời vòng vo né tránh câu hỏi.""",
    },
    {
        "key": "faithfulness",
        "label": "Faithfulness",
        "title": "Faithfulness / Anti-Hallucination (Tính trung thực & Đo lường ảo giác)",
        "rubric": """Faithfulness / Anti-Hallucination (Tính trung thực & Đo lường ảo giác) [0.0 - 10.0]:
   - Đánh giá mức độ thông tin trong câu trả lời hoàn toàn xuất phát và có căn cứ từ Retrieved Chunks.
   - 10.0: Hoàn toàn trung thực, 100% luận điểm đều có căn cứ từ tài liệu được trích xuất (hoặc Reference). Không có bất kỳ ảo giác (hallucination) nào.
   - 7.0 - 9.0: Phần lớn bám sát tài liệu, có một vài suy diễn logic hợp lý.
   - 4.0 - 6.0: Xuất hiện các thông tin tự bịa hoặc không có căn cứ trong tài liệu.
   - 0.0 - 3.0: Ảo giác nghiêm trọng (Severe Hallucination), bịa đặt điều luật, số liệu hoặc sự kiện hoàn toàn không tồn tại trong ngữ cảnh.""",
    },
]


def calculate_keyword_coverage(reference: str, used_chunks: list[str]) -> float:
    """Tính tỷ lệ bao phủ từ khóa cơ bản từ Reference xuất hiện trong các chunks.

    LƯU Ý (giữ nguyên để so sánh lịch sử với các lần chạy trước / README):
    đây là công thức GỐC, dùng SUBSTRING match (`w in all_chunks_text`) chứ
    không phải khớp theo ranh giới từ. Hệ quả: điểm số này tăng gần như đơn
    điệu theo SỐ LƯỢNG chunk, không phản ánh chất lượng — 40 chunk NGẪU
    NHIÊN lấy từ corpus đã đạt ~86.5% theo công thức này (đo thực nghiệm).
    Dùng `calculate_keyword_coverage_strict` để có con số đáng tin hơn; giữ
    cả hai để không làm mất khả năng đối chiếu với điểm cũ.
    """
    if not reference:
        return 100.0
    if not used_chunks:
        return 0.0

    all_chunks_text = " ".join(used_chunks).lower()
    words = re.findall(r"\b\w{3,}\b", reference.lower())
    if not words:
        return 100.0

    unique_words = list(set(words))
    matched = sum(1 for w in unique_words if w in all_chunks_text)
    coverage = (matched / len(unique_words)) * 100.0
    return min(100.0, coverage)


# Stopword tiếng Việt tối thiểu — chỉ các hư từ/liên từ cực phổ biến, không
# lọc từ có nghĩa chuyên ngành. Mục tiêu là loại các từ nối gần như luôn
# match dù chunk không liên quan (vd. "của", "là", "và"), không phải làm
# một bộ NLP stopword đầy đủ.
_VN_STOPWORDS = {
    "và",
    "của",
    "là",
    "có",
    "được",
    "cho",
    "các",
    "một",
    "những",
    "này",
    "đó",
    "khi",
    "với",
    "để",
    "trong",
    "về",
    "theo",
    "như",
    "từ",
    "hay",
    "hoặc",
    "thì",
    "nếu",
    "nên",
    "mà",
    "đã",
    "sẽ",
    "đang",
    "bị",
    "do",
    "vì",
    "nhưng",
    "còn",
    "cũng",
    "chỉ",
    "rất",
    "vào",
    "ra",
    "trên",
    "dưới",
    "giữa",
    "sau",
    "trước",
    "tại",
    "bởi",
    "cả",
    "mỗi",
    "tất",
}


def calculate_keyword_coverage_strict(reference: str, used_chunks: list[str]) -> float:
    """
    Bản coverage ĐÁNG TIN CẬY hơn: khớp theo RANH GIỚI TỪ (token-level, không
    phải substring), bỏ stopword tiếng Việt phổ biến, tối thiểu 2 ký tự/từ.

    So với `calculate_keyword_coverage`: substring match khiến số chunk càng
    nhiều thì điểm càng cao dù nội dung không liên quan (đo thực nghiệm: 40
    chunk ngẫu nhiên từ corpus đạt 86.5%). Token-set match không có hiệu ứng
    đó — thêm chunk không liên quan không làm tăng số từ khớp.
    """
    if not reference:
        return 100.0
    if not used_chunks:
        return 0.0

    chunk_tokens: set[str] = set()
    for c in used_chunks:
        chunk_tokens.update(re.findall(r"\b\w{2,}\b", c.lower()))

    ref_words = [
        w
        for w in re.findall(r"\b\w{2,}\b", reference.lower())
        if w not in _VN_STOPWORDS
    ]
    if not ref_words:
        return 100.0

    unique_words = set(ref_words)
    matched = sum(1 for w in unique_words if w in chunk_tokens)
    coverage = (matched / len(unique_words)) * 100.0
    return min(100.0, coverage)


def parse_metric_output(raw_text: str) -> Optional[MetricEval]:
    """Trích xuất và parse an toàn kết quả chấm 1 tiêu chí từ phản hồi LLM (JSON hoặc Regex)."""
    if not raw_text or not raw_text.strip():
        return None

    # Tier 1: Parse JSON block
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match2 = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            json_str = match2.group(1) if match2 else raw_text
        data = json.loads(json_str)
        return MetricEval(
            score=min(10.0, max(0.0, float(data.get("score", 0.0)))),
            feedback=str(data.get("feedback", "")).strip(),
        )
    except Exception:
        pass

    # Tier 2: Trích xuất key-value bằng Regex
    try:
        score_m = re.search(
            r'["\']?score["\']?\s*[:=]\s*([0-9.]+)', raw_text, re.IGNORECASE
        )
        fb_m = re.search(
            r'["\']?feedback["\']?\s*[:=]\s*["\']?((?:[^"\'\\]|\\.)*)',
            raw_text,
            re.IGNORECASE,
        )

        if score_m:
            fb_text = (
                fb_m.group(1).encode().decode("unicode_escape", errors="ignore")
                if fb_m
                else ""
            )
            return MetricEval(
                score=min(10.0, max(0.0, float(score_m.group(1)))),
                feedback=fb_text.strip()
                or "(Đã trích xuất điểm số thành công qua Regex)",
            )
    except Exception:
        pass

    return None


async def evaluate_single_metric_with_judge(
    llm,
    fallback_llm,
    criterion: dict,
    question: str,
    reference: str,
    response: str,
    used_chunks: list[str],
) -> Optional[MetricEval]:
    """
    Chấm điểm MỘT tiêu chí duy nhất (accuracy / completeness / relevance /
    faithfulness) bằng 1 lượt gọi LLM Judge riêng — không lẫn rubric của các
    tiêu chí khác — với cơ chế Fallback và rút gọn ngữ cảnh an toàn.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    chunks_str = json.dumps(used_chunks, ensure_ascii=False) if used_chunks else "[]"
    chunks_str = _collapse_pathological_repetition(chunks_str)
    if len(chunks_str) > _MAX_EVAL_CHUNKS_CHARS:
        chunks_str = (
            chunks_str[:_MAX_EVAL_CHUNKS_CHARS]
            + "\n... [ ngữ cảnh được rút gọn an toàn ]"
        )

    ref_str = reference if reference else "(Không có đáp án mẫu)"
    ref_str = _collapse_pathological_repetition(ref_str)
    if len(ref_str) > _MAX_EVAL_REF_CHARS:
        ref_str = (
            ref_str[:_MAX_EVAL_REF_CHARS] + "\n... [ đáp án mẫu được rút gọn an toàn ]"
        )

    ans_str = response if response else "(Không có câu trả lời)"
    # Rút gọn đoạn lặp bất thường (generate_node degenerate output) TRƯỚC khi
    # cắt theo độ dài — nếu không, giới hạn ký tự phía trên có thể cắt trúng
    # ngay giữa vùng rác lặp, khiến judge tưởng câu trả lời hợp lệ bị cắt dở.
    ans_str = _collapse_pathological_repetition(ans_str)
    if len(ans_str) > _MAX_EVAL_ANS_CHARS:
        ans_str = (
            ans_str[:_MAX_EVAL_ANS_CHARS] + "\n... [ câu trả lời được rút gọn an toàn ]"
        )

    user_prompt = f"""Câu hỏi (Question):
{question}

Đáp án tham chiếu chuẩn (Reference Answer):
{ref_str}

Câu trả lời của hệ thống (Generated Answer):
{ans_str}

Các đoạn ngữ cảnh tài liệu đã sử dụng (Retrieved Chunks):
{chunks_str}

Tiêu chí DUY NHẤT cần chấm — {criterion["title"]}:
{criterion["rubric"]}

Hãy chấm điểm theo đúng thang điểm trên (0.0 đến 10.0) và viết feedback ngắn gọn (dưới 80 từ) CHỈ về tiêu chí này.
Trả về JSON duy nhất có cấu trúc:
{{
  "score": 9.5,
  "feedback": "Nhận xét ngắn gọn riêng cho tiêu chí này..."
}}"""

    messages = [
        SystemMessage(content=JUDGE_PREAMBLE),
        HumanMessage(content=user_prompt),
    ]

    if llm:
        try:
            res = await llm.ainvoke(messages)
            parsed = parse_metric_output(res.content)
            if parsed is not None:
                return parsed
            logger.warning(
                f"[Judge:{criterion['key']}] Primary LLM trả về format không hợp lệ, thử chuyển Fallback..."
            )
        except Exception as e:
            logger.warning(
                f"[Judge:{criterion['key']}] Primary LLM thất bại: {e}. Đang chuyển sang Fallback..."
            )

    if fallback_llm:
        try:
            res = await fallback_llm.ainvoke(messages)
            parsed = parse_metric_output(res.content)
            if parsed is not None:
                return parsed
            logger.error(
                f"[Judge:{criterion['key']}] Fallback LLM parse thất bại: {repr(res.content[:200])}"
            )
        except Exception as e:
            logger.error(f"[Judge:{criterion['key']}] Fallback LLM thất bại: {e}")

    return None


async def evaluate_single_answer_with_judge(
    llm,
    fallback_llm,
    question: str,
    reference: str,
    response: str,
    used_chunks: list[str],
) -> Optional[AnswerEval]:
    """
    Đánh giá một câu trả lời bằng LLM Judge — chấm 4 tiêu chí bằng 4 PROMPT
    ĐỘC LẬP, chạy song song (asyncio.gather), rồi tổng hợp lại thành 1
    AnswerEval. Tách prompt để tránh halo effect giữa các tiêu chí (xem
    docstring AnswerEval); chạy song song để tổng thời gian chấm 1 câu không
    tăng so với khi dùng 1 prompt duy nhất (chỉ tăng số lượt gọi LLM/chi phí).

    Nếu BẤT KỲ tiêu chí nào thất bại (cả primary lẫn fallback đều không parse
    được), toàn bộ câu bị đánh dấu thất bại — giữ đúng ngữ nghĩa cũ (loại bỏ
    câu đánh giá thất bại khỏi điểm tổng), tránh tính điểm trung bình từ một
    hỗn hợp tiêu chí có/không có kết quả.
    """
    metric_results = await asyncio.gather(
        *[
            evaluate_single_metric_with_judge(
                llm, fallback_llm, criterion, question, reference, response, used_chunks
            )
            for criterion in CRITERIA
        ]
    )

    if any(r is None for r in metric_results):
        return None

    scores = {c["key"]: r.score for c, r in zip(CRITERIA, metric_results)}
    feedback = "\n".join(
        f"**{c['label']}**: {r.feedback}" for c, r in zip(CRITERIA, metric_results)
    )

    return AnswerEval(
        accuracy=scores["accuracy"],
        completeness=scores["completeness"],
        relevance=scores["relevance"],
        faithfulness=scores["faithfulness"],
        feedback=feedback,
    )


def _parse_trajectory_summary(
    traj_text: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Trích 3 chỉ số ngắn gọn từ text trajectory (dùng chung cho cả câu vừa
    chạy lẫn câu nạp từ cache — cả hai đều ở dạng text render sẵn, nên chỉ
    cần parse text thay vì phải giữ lại Trajectory object).

    "Vòng lặp retrieve: N/MAX" giữ nguyên ý nghĩa MAX qua các facet (xem
    Trajectory.retrieve_count trong agent/observability/trajectory.py) —
    KHÔNG đổi format dòng này để không vỡ script/CSV cũ đang parse nó.
    "Vòng lặp theo facet: label1=n1, label2=n2, ..." là dòng MỚI, chỉ xuất
    hiện khi trajectory có >= 1 facet (mọi trajectory mới đều có).

    Returns:
        (traj_loops: "2/3" hoặc None,
         traj_tools: "semantic_search x1, ..." hoặc None,
         traj_facet_loops: "label1=n1, label2=n2, ..." hoặc None)
    """
    if not traj_text:
        return None, None, None
    loop_match = re.search(r"Vòng lặp retrieve:\s*(\d+/\d+)", traj_text)
    tools_match = re.search(r"Tool đã gọi\s*:\s*(.+)", traj_text)
    facet_loops_match = re.search(r"Vòng lặp theo facet:\s*(.+)", traj_text)
    traj_loops = loop_match.group(1) if loop_match else None
    traj_tools = tools_match.group(1).strip() if tools_match else None
    traj_facet_loops = facet_loops_match.group(1).strip() if facet_loops_match else None
    return traj_loops, traj_tools, traj_facet_loops


def run_evaluation_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict, bool]:
    """
    Thực hiện đánh giá toàn diện một chủ đề.
    Nếu một câu hỏi bị lỗi (không đánh giá được), câu đó sẽ được đánh dấu FAILED
    và KHÔNG ĐƯỢC TÍNH VÀO ĐIỂM TỔNG để tránh làm lệch kết quả.

    Returns:
        (detail_df, summary_dict, was_cached)
    """
    results, was_cached = get_answers_and_chunks_for_topic(
        topic, force_rerun=force_rerun, progress_callback=progress_callback
    )

    primary_llm, fallback_llm = create_evaluator_llm()
    if not primary_llm and not fallback_llm:
        raise RuntimeError(
            "Không có LLM Judge nào khả dụng. Vui lòng kiểm tra LLM_BASE_URL hoặc GEMINI_API_KEY trong .env"
        )

    rows = []
    total = len(results)

    for idx, item in enumerate(results, 1):
        num = item["num"]
        q_text = item["question"]
        resp = item["response"]
        ref = item["reference"]
        chunks = item["used_chunks"]
        traj_text = item.get("trajectory")
        traj_loops, traj_tools, traj_facet_loops = _parse_trajectory_summary(traj_text)

        if progress_callback:
            progress_callback(f"[{topic}] LLM Judge đang chấm câu {idx}/{total}...")

        logger.info(f"[{topic}] Đang chấm câu {idx}/{total}...")

        eval_res = asyncio.run(
            evaluate_single_answer_with_judge(
                primary_llm, fallback_llm, q_text, ref, resp, chunks
            )
        )

        coverage = calculate_keyword_coverage(ref, chunks)
        coverage_strict = calculate_keyword_coverage_strict(ref, chunks)

        if eval_res is not None:
            is_success = True
            acc_val = round(eval_res.accuracy, 2)
            comp_val = round(eval_res.completeness, 2)
            rel_val = round(eval_res.relevance, 2)
            faith_val = round(eval_res.faithfulness, 2)
            total_score = round((acc_val + comp_val + rel_val + faith_val) / 4.0, 2)
            feedback_val = eval_res.feedback
        else:
            is_success = False
            acc_val = None
            comp_val = None
            rel_val = None
            faith_val = None
            total_score = None
            feedback_val = (
                "[LỖI ĐÁNH GIÁ] Không nhận được phản hồi hợp lệ từ LLM Judge."
            )

        rows.append(
            {
                "STT": num,
                "topic": topic,
                "question": q_text,
                "response": resp,
                "reference": ref,
                "accuracy": acc_val,
                "completeness": comp_val,
                "relevance": rel_val,
                "faithfulness": faith_val,
                "total_score": total_score,
                "coverage_pct": round(coverage, 1),
                "coverage_pct_strict": round(coverage_strict, 1),
                "chunks_count": len(chunks),
                "feedback": feedback_val,
                "is_success": is_success,
                "used_chunks": chunks,
                "traj_loops": traj_loops,
                "traj_tools": traj_tools,
                "traj_facet_loops": traj_facet_loops,
                "traj_text": traj_text
                or "(chưa có trajectory — cache cũ trước khi bật tính năng này)",
            }
        )

    detail_df = pd.DataFrame(rows)

    valid_df = detail_df[detail_df["is_success"] == True]
    success_count = len(valid_df)
    failed_count = len(detail_df) - success_count

    if not valid_df.empty:
        summary = {
            "accuracy": float(valid_df["accuracy"].astype(float).mean()),
            "completeness": float(valid_df["completeness"].astype(float).mean()),
            "relevance": float(valid_df["relevance"].astype(float).mean()),
            "faithfulness": float(valid_df["faithfulness"].astype(float).mean()),
            "total_score": float(valid_df["total_score"].astype(float).mean()),
            "coverage_pct": float(valid_df["coverage_pct"].astype(float).mean()),
            "coverage_pct_strict": float(
                valid_df["coverage_pct_strict"].astype(float).mean()
            ),
            "avg_chunks": float(detail_df["chunks_count"].mean()),
            "total_questions": len(detail_df),
            "success_count": success_count,
            "failed_count": failed_count,
        }
    else:
        summary = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "faithfulness": 0.0,
            "total_score": 0.0,
            "coverage_pct": 0.0,
            "coverage_pct_strict": 0.0,
            "avg_chunks": 0.0,
            "total_questions": len(detail_df),
            "success_count": 0,
            "failed_count": failed_count,
        }

    return detail_df, summary, was_cached


# =============================================================================
# 6. HTML KPI CARDS & VISUALIZATION
# =============================================================================


def get_color(value: Optional[float], is_percentage: bool = False) -> str:
    """Xác định màu sắc theo ngưỡng điểm (Thang 10 hoặc %)."""
    if value is None:
        return "#94a3b8"
    if is_percentage:
        if value >= COVERAGE_GREEN:
            return "#10b981"
        elif value >= COVERAGE_AMBER:
            return "#f59e0b"
        else:
            return "#ef4444"
    else:
        if value >= SCORE_GREEN:
            return "#10b981"
        elif value >= SCORE_AMBER:
            return "#f59e0b"
        else:
            return "#ef4444"


def format_metric_html(
    label: str,
    value: Optional[float],
    is_percentage: bool = False,
    is_count: bool = False,
) -> str:
    """Tạo khối thẻ HTML KPI card với viền màu động sang trọng."""
    if value is None:
        color = "#94a3b8"
        val_str = "N/A"
    elif is_count:
        color = "#3b82f6"
        val_str = f"{value:.1f} chunks"
    elif is_percentage:
        color = get_color(value, is_percentage=True)
        val_str = f"{value:.1f}%"
    else:
        color = get_color(value, is_percentage=False)
        val_str = f"{value:.2f}/10"

    return f"""
    <div style="margin: 6px 0; padding: 12px 16px; background-color: #f8fafc; border-radius: 8px; border-left: 6px solid {color}; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>
        <div style="font-size: 24px; font-weight: 800; color: {color}; line-height: 1.1;">{val_str}</div>
    </div>
    """


def generate_summary_kpi_html(
    summary: dict, topic_name: str = "", was_cached: bool = False
) -> str:
    """Tạo toàn bộ khối KPI metrics HTML cho Dashboard."""
    cache_badge = (
        '<span style="background-color: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">⚡ Cache (answer.txt & chunks.txt)</span>'
        if was_cached
        else '<span style="background-color: #fef3c7; color: #b45309; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">🤖 Live Agent Run</span>'
    )

    failed_cnt = summary.get("failed_count", 0)
    failed_badge = (
        f' | <span style="color: #ef4444; font-weight: 700;">⚠️ {failed_cnt} câu lỗi (không tính vào điểm)</span>'
        if failed_cnt > 0
        else ""
    )

    return f"""
    <div style="padding: 10px 0;">
        <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 15px; font-weight: 700; color: #1e293b;">📊 Tổng quan đánh giá: {topic_name}</span>
            {cache_badge}
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
            {format_metric_html("🏆 Tổng điểm (Total Score)", summary.get("total_score"))}
            {format_metric_html("🎯 Độ chính xác (Accuracy)", summary.get("accuracy"))}
            {format_metric_html("📚 Độ đầy đủ (Completeness)", summary.get("completeness"))}
            {format_metric_html("🎯 Độ liên quan (Relevance)", summary.get("relevance"))}
            {format_metric_html("🛡️ Trung thực (Faithfulness)", summary.get("faithfulness"))}
            {format_metric_html("🔍 Phủ từ khóa (Keyword Coverage - legacy)", summary.get("coverage_pct"), is_percentage=True)}
            {format_metric_html("🔍 Phủ từ khóa (Keyword Coverage - strict)", summary.get("coverage_pct_strict"), is_percentage=True)}
        </div>
        <div style="margin-top: 10px; padding: 8px 12px; background-color: #f1f5f9; border-radius: 6px; font-size: 12px; color: #475569; text-align: center;">
            Tổng số câu hỏi: <b>{summary.get("total_questions", 0)}</b> (Thành công: <b>{summary.get("success_count", 0)}</b>{failed_badge}) | Trung bình Context: <b>{summary.get("avg_chunks", 0.0):.1f} chunks/câu</b>
        </div>
    </div>
    """


def create_single_topic_chart(summary: dict, topic: str) -> plt.Figure:
    """Tạo biểu đồ cột thể hiện các chỉ số của một chủ đề."""
    metrics = ["Accuracy", "Completeness", "Relevance", "Faithfulness", "Total Score"]
    scores = [
        summary.get("accuracy", 0.0),
        summary.get("completeness", 0.0),
        summary.get("relevance", 0.0),
        summary.get("faithfulness", 0.0),
        summary.get("total_score", 0.0),
    ]

    colors = [get_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    bars = ax.bar(
        metrics, scores, color=colors, width=0.55, edgecolor="white", linewidth=1.5
    )

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            color="#1e293b",
        )

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm (Thang 10)", fontsize=11, fontweight="600")
    ax.set_title(
        f"Phân tích chất lượng RAG — Chủ đề: {topic}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.axhline(
        y=SCORE_GREEN,
        color="#10b981",
        linestyle="--",
        alpha=0.6,
        label="Mục tiêu (>= 8.5)",
    )
    ax.axhline(
        y=SCORE_AMBER,
        color="#f59e0b",
        linestyle=":",
        alpha=0.6,
        label="Ngưỡng đạt (>= 7.0)",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_all_topics_chart(all_summaries: dict[str, dict]) -> plt.Figure:
    """Tạo biểu đồ so sánh các chủ đề."""
    topics = list(all_summaries.keys())
    if not topics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        return fig

    x = np.arange(len(topics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(topics) * 2.5), 5), dpi=100)

    acc = [all_summaries[t].get("accuracy", 0.0) for t in topics]
    comp = [all_summaries[t].get("completeness", 0.0) for t in topics]
    rel = [all_summaries[t].get("relevance", 0.0) for t in topics]
    faith = [all_summaries[t].get("faithfulness", 0.0) for t in topics]

    ax.bar(
        x - 1.5 * width,
        acc,
        width,
        label="Accuracy",
        color="#3b82f6",
        edgecolor="white",
    )
    ax.bar(
        x - 0.5 * width,
        comp,
        width,
        label="Completeness",
        color="#8b5cf6",
        edgecolor="white",
    )
    ax.bar(
        x + 0.5 * width,
        rel,
        width,
        label="Relevance",
        color="#06b6d4",
        edgecolor="white",
    )
    ax.bar(
        x + 1.5 * width,
        faith,
        width,
        label="Faithfulness",
        color="#10b981",
        edgecolor="white",
    )

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm (Thang 10)", fontsize=11, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, fontsize=11, fontweight="600")
    ax.set_title(
        "So sánh các chỉ số RAG giữa các chủ đề", fontsize=14, fontweight="bold", pad=12
    )
    ax.axhline(
        y=SCORE_GREEN,
        color="#10b981",
        linestyle="--",
        alpha=0.5,
        label="Mục tiêu (8.5)",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_summary_table(all_summaries: dict[str, dict]) -> pd.DataFrame:
    """Tạo bảng tổng hợp điểm cho các chủ đề."""
    rows = []
    for topic, summary in all_summaries.items():
        rows.append(
            {
                "Chủ đề (Topic)": topic,
                "Accuracy": round(summary.get("accuracy", 0.0), 2),
                "Completeness": round(summary.get("completeness", 0.0), 2),
                "Relevance": round(summary.get("relevance", 0.0), 2),
                "Faithfulness": round(summary.get("faithfulness", 0.0), 2),
                "Total Score": round(summary.get("total_score", 0.0), 2),
                "Keyword Coverage legacy (%)": f"{summary.get('coverage_pct', 0.0):.1f}%",
                "Keyword Coverage strict (%)": f"{summary.get('coverage_pct_strict', 0.0):.1f}%",
                "Avg Chunks": round(summary.get("avg_chunks", 0.0), 1),
                "Số câu thành công": f"{summary.get('success_count', 0)}/{summary.get('total_questions', 0)}",
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 7. GRADIO APP & HANDLERS
# =============================================================================

_current_detail_df: Optional[pd.DataFrame] = None


def export_evaluation_results(
    topic: str, detail_df: pd.DataFrame, summary: dict
) -> tuple[Path, Path]:
    """
    Ghi kết quả đánh giá ra file, để có baseline đối chiếu qua các lần chạy
    (trước đây mọi điểm số chỉ tồn tại trong _current_detail_df trong bộ nhớ,
    mất khi tắt process — không cách nào so sánh trước/sau khi sửa agent).

    Ghi 2 file trong data/test/<topic>/:
        scores_<timestamp>.csv  — chi tiết từng câu (không kèm cột list dài
                                  used_chunks/traj_text để file gọn, xem lại
                                  trong chunks.txt/trajectory.txt tương ứng).
        summary_<timestamp>.json — điểm tổng hợp của lần chạy này.

    Returns:
        (csv_path, json_path)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_TEST_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"scores_{ts}.csv"
    json_path = out_dir / f"summary_{ts}.json"

    export_cols = [
        c for c in detail_df.columns if c not in ("used_chunks", "traj_text")
    ]
    detail_df[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"topic": topic, "timestamp": ts, **summary},
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"[{topic}] Đã xuất kết quả: {csv_path.name}, {json_path.name}")
    return csv_path, json_path


def evaluate_single_topic_handler(
    topic: str, force_rerun: bool, progress=gr.Progress()
):
    """Handler đánh giá một chủ đề."""
    global _current_detail_df

    if not topic:
        return (
            None,
            None,
            "<p>⚠️ Vui lòng chọn một chủ đề!</p>",
            gr.Dropdown(choices=[]),
            "⚠️ Vui lòng chọn một chủ đề!",
        )

    try:
        progress(0.1, desc=f"Đang chuẩn bị dữ liệu cho chủ đề: {topic}...")

        def cb(msg):
            progress(0.4, desc=msg)

        detail_df, summary, was_cached = run_evaluation_for_topic(
            topic, force_rerun=force_rerun, progress_callback=cb
        )

        _current_detail_df = detail_df

        progress(0.85, desc="Đang tạo biểu đồ & báo cáo...")

        kpi_html = generate_summary_kpi_html(
            summary, topic_name=topic, was_cached=was_cached
        )
        chart = create_single_topic_chart(summary, topic)

        display_cols = [
            "STT",
            "question",
            "response",
            "accuracy",
            "completeness",
            "relevance",
            "faithfulness",
            "total_score",
            "coverage_pct",
            "coverage_pct_strict",
            "chunks_count",
            "feedback",
        ]
        display_df = detail_df[display_cols].copy()
        display_df = display_df.fillna("-")

        export_evaluation_results(topic, detail_df, summary)

        q_choices = [
            f"Câu {row['STT']}: {str(row['question'])[:60]}..."
            for _, row in detail_df.iterrows()
        ]
        first_choice = q_choices[0] if q_choices else None

        succ = summary.get("success_count", 0)
        tot = summary.get("total_questions", 0)
        fail = summary.get("failed_count", 0)
        fail_msg = f" (⚠️ {fail} câu lỗi không tính vào điểm tổng)" if fail > 0 else ""
        status_msg = f"✅ Hoàn tất đánh giá chủ đề **{topic}** (Thành công {succ}/{tot} câu{fail_msg})!"

        progress(1.0, desc="Hoàn tất!")

        return (
            chart,
            display_df,
            kpi_html,
            gr.Dropdown(choices=q_choices, value=first_choice),
            status_msg,
        )

    except Exception as e:
        logger.error(f"Lỗi đánh giá {topic}: {e}", exc_info=True)
        err_msg = f"❌ Lỗi: {e}\n\n{traceback.format_exc()}"
        return (
            None,
            None,
            f"<div style='color: red;'>{err_msg}</div>",
            gr.Dropdown(choices=[]),
            err_msg,
        )


def inspect_question_handler(selected_q_text: str):
    """Handler hiển thị chi tiết câu hỏi, đáp án, chunks, feedback và trajectory."""
    global _current_detail_df
    if _current_detail_df is None or not selected_q_text:
        return "", "", "", ""

    match = re.match(r"^Câu\s*(\d+):", selected_q_text)
    if not match:
        return "", "", "", ""

    stt = int(match.group(1))
    row = _current_detail_df[_current_detail_df["STT"] == stt]
    if row.empty:
        return "", "", "", ""

    item = row.iloc[0]

    col1_content = f"""### ❓ Câu hỏi #{item["STT"]}
**{item["question"]}**

---
### 📖 Đáp án mẫu (Reference Answer)
{item["reference"] if item["reference"] else "*(Không có đáp án mẫu)*"}

---
### 🤖 Câu trả lời của CRAG Agent
{item["response"]}
"""

    chunks_list = item.get("used_chunks", [])
    if not chunks_list:
        chunks_content = "*(Không có context chunk nào được sử dụng)*"
    else:
        chunks_blocks = []
        for c_idx, chunk_text in enumerate(chunks_list, 1):
            chunks_blocks.append(f"**[Chunk {c_idx}]**\n```markdown\n{chunk_text}\n```")
        chunks_content = "\n\n".join(chunks_blocks)

    col2_content = f"""### 📄 Các Chunks tài liệu cuối cùng đã dùng ({len(chunks_list)} chunks)
{chunks_content}
"""

    acc_str = f"{item['accuracy']}/10" if pd.notna(item["accuracy"]) else "N/A"
    comp_str = f"{item['completeness']}/10" if pd.notna(item["completeness"]) else "N/A"
    rel_str = f"{item['relevance']}/10" if pd.notna(item["relevance"]) else "N/A"
    faith_str = (
        f"{item['faithfulness']}/10" if pd.notna(item["faithfulness"]) else "N/A"
    )
    tot_str = f"{item['total_score']}/10" if pd.notna(item["total_score"]) else "N/A"

    col3_content = f"""### ⚖️ Đánh giá từ LLM Judge (Thang 10)
- **Độ chính xác (Accuracy)**: `{acc_str}`
- **Độ đầy đủ (Completeness)**: `{comp_str}`
- **Độ liên quan (Relevance)**: `{rel_str}`
- **Trung thực / Chống ảo giác (Faithfulness)**: `{faith_str}`
- **🏆 Tổng điểm (Total Score)**: `{tot_str}`
- **Độ phủ từ khóa - legacy (Keyword Coverage)**: `{item["coverage_pct"]}%`
- **Độ phủ từ khóa - strict (Keyword Coverage)**: `{item.get("coverage_pct_strict", "N/A")}%`

---
### 📝 Nhận xét chi tiết (Feedback)
{item["feedback"]}
"""

    traj_text = item.get("traj_text") or "*(Không có trajectory)*"
    col4_content = f"""### 🧭 Trajectory (quỹ đạo thực thi Agent)
```
{traj_text}
```
"""

    return col1_content, col2_content, col3_content, col4_content


def evaluate_all_topics_handler(force_rerun: bool, progress=gr.Progress()):
    """Handler đánh giá toàn bộ các chủ đề."""
    topics = get_available_topics()
    if not topics:
        return (
            None,
            None,
            None,
            "<p>⚠️ Không tìm thấy chủ đề nào trong data/test/</p>",
            "⚠️ Không có dữ liệu.",
        )

    all_summaries = {}
    all_details = []
    total_questions = 0
    total_success = 0
    total_failed = 0

    for t_idx, topic in enumerate(topics):
        try:
            base_prog = t_idx / len(topics)
            progress(
                base_prog, desc=f"[{t_idx + 1}/{len(topics)}] Đang xử lý: {topic}..."
            )

            def cb(msg):
                progress(base_prog + 0.5 / len(topics), desc=msg)

            detail_df, summary, was_cached = run_evaluation_for_topic(
                topic, force_rerun=force_rerun, progress_callback=cb
            )

            all_summaries[topic] = summary
            all_details.append(detail_df)
            total_questions += summary.get("total_questions", 0)
            total_success += summary.get("success_count", 0)
            total_failed += summary.get("failed_count", 0)

            export_evaluation_results(topic, detail_df, summary)

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá chủ đề {topic}: {e}", exc_info=True)
            all_summaries[topic] = {
                "accuracy": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "faithfulness": 0.0,
                "total_score": 0.0,
                "coverage_pct": 0.0,
                "coverage_pct_strict": 0.0,
                "avg_chunks": 0.0,
                "total_questions": 0,
                "success_count": 0,
                "failed_count": 0,
                "error": str(e),
            }

    progress(0.9, desc="Đang tạo biểu đồ tổng hợp...")

    chart = create_all_topics_chart(all_summaries)
    summary_df = create_summary_table(all_summaries)

    if all_details:
        combined_df = pd.concat(all_details, ignore_index=True)
        display_cols = [
            "STT",
            "topic",
            "question",
            "response",
            "accuracy",
            "completeness",
            "relevance",
            "faithfulness",
            "total_score",
            "coverage_pct",
            "coverage_pct_strict",
            "chunks_count",
            "feedback",
        ]
        detail_table_df = combined_df[
            [c for c in display_cols if c in combined_df.columns]
        ].copy()
        detail_table_df = detail_table_df.fillna("-")
    else:
        detail_table_df = pd.DataFrame()

    valid_summaries = [
        s
        for s in all_summaries.values()
        if "error" not in s and s.get("success_count", 0) > 0
    ]
    if valid_summaries:
        overall_summary = {
            "total_score": sum(s["total_score"] for s in valid_summaries)
            / len(valid_summaries),
            "accuracy": sum(s["accuracy"] for s in valid_summaries)
            / len(valid_summaries),
            "completeness": sum(s["completeness"] for s in valid_summaries)
            / len(valid_summaries),
            "relevance": sum(s["relevance"] for s in valid_summaries)
            / len(valid_summaries),
            "faithfulness": sum(s["faithfulness"] for s in valid_summaries)
            / len(valid_summaries),
            "coverage_pct": sum(s["coverage_pct"] for s in valid_summaries)
            / len(valid_summaries),
            "coverage_pct_strict": sum(
                s.get("coverage_pct_strict", 0.0) for s in valid_summaries
            )
            / len(valid_summaries),
            "avg_chunks": sum(s["avg_chunks"] for s in valid_summaries)
            / len(valid_summaries),
            "total_questions": total_questions,
            "success_count": total_success,
            "failed_count": total_failed,
        }
    else:
        overall_summary = {}

    kpi_html = generate_summary_kpi_html(
        overall_summary,
        topic_name="Toàn hệ thống (All Topics)",
        was_cached=not force_rerun,
    )

    fail_msg = (
        f" (⚠️ {total_failed} câu lỗi không tính vào điểm tổng)"
        if total_failed > 0
        else ""
    )
    status_msg = f"✅ Đã hoàn tất đánh giá {len(valid_summaries)}/{len(topics)} chủ đề (Thành công {total_success}/{total_questions} câu{fail_msg})!"

    progress(1.0, desc="Hoàn tất!")

    return chart, summary_df, detail_table_df, kpi_html, status_msg


# =============================================================================
# 8. GIAO DIỆN GRADIO (THEME & BLOCKS)
# =============================================================================


def build_gradio_app():
    """Khởi tạo giao diện Dashboard Gradio."""
    topics = get_available_topics()
    topic_choices = topics if topics else ["(Không có chủ đề)"]

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        font=["Inter", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="SLIDEGEN-RAG Evaluation Dashboard") as app:
        gr.Markdown(
            """
            # 📊 SLIDEGEN-RAG — Dashboard Đánh Giá Chất Lượng
            Hệ thống đánh giá toàn diện **CRAG Agent** sử dụng phương pháp **LLM-as-a-Judge** trực tiếp (Thang điểm 10).
            
            **Các tiêu chí đánh giá cốt lõi:**
            - 🎯 **Accuracy (Độ chính xác)**: So sánh với đáp án mẫu, kiểm tra tính đúng đắn của thông tin.
            - 📚 **Completeness (Độ đầy đủ)**: Bao quát toàn bộ các ý cốt lõi và căn cứ trong tài liệu.
            - 🎯 **Relevance (Độ liên quan)**: Bám sát câu hỏi, không lan man, không chứa thông tin thừa.
            - 🛡️ **Faithfulness (Chống ảo giác / Hallucination)**: Đảm bảo câu trả lời chỉ xuất phát từ các context chunks đã sử dụng.
            """
        )

        with gr.Tabs():
            # ─────────────────────────────────────────────────────────────────
            # TAB 1: Đánh giá theo chủ đề
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎯 Đánh giá theo chủ đề", id="tab_single"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=2):
                        topic_dropdown = gr.Dropdown(
                            choices=topic_choices,
                            value=topic_choices[0] if topic_choices else None,
                            label="Chọn chủ đề kiểm thử (Topic)",
                            interactive=True,
                        )
                    with gr.Column(scale=2):
                        chk_force_single = gr.Checkbox(
                            label="🔄 Buộc chạy lại Agent (Bỏ qua cache answer/chunks)",
                            value=False,
                        )
                    with gr.Column(scale=1):
                        btn_eval_single = gr.Button(
                            "🚀 Bắt đầu đánh giá",
                            variant="primary",
                            size="lg",
                        )

                status_single = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_single = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Bắt đầu đánh giá' để hiển thị các chỉ số</div>"
                        )
                    with gr.Column(scale=1):
                        chart_single = gr.Plot(label="Biểu đồ phân tích chất lượng")

                gr.Markdown("### 📋 Bảng điểm chi tiết từng câu hỏi")
                table_single = gr.Dataframe(
                    label="Chi tiết kết quả đánh giá",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔍 Soi chi tiết từng câu hỏi (Question Inspector)")
                with gr.Row():
                    inspect_dropdown = gr.Dropdown(
                        choices=[],
                        label="Chọn câu hỏi cần soi chi tiết",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_col1 = gr.Markdown(
                            "*(Chọn một câu hỏi ở trên để xem chi tiết)*"
                        )
                    with gr.Column(scale=1):
                        inspect_col2 = gr.Markdown("")
                    with gr.Column(scale=1):
                        inspect_col3 = gr.Markdown("")
                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_col4 = gr.Markdown(
                            label="Trajectory (quỹ đạo thực thi Agent)",
                        )

                btn_eval_single.click(
                    fn=evaluate_single_topic_handler,
                    inputs=[topic_dropdown, chk_force_single],
                    outputs=[
                        chart_single,
                        table_single,
                        kpi_single,
                        inspect_dropdown,
                        status_single,
                    ],
                )

                inspect_dropdown.change(
                    fn=inspect_question_handler,
                    inputs=[inspect_dropdown],
                    outputs=[inspect_col1, inspect_col2, inspect_col3, inspect_col4],
                )

            # ─────────────────────────────────────────────────────────────────
            # TAB 2: Đánh giá tất cả chủ đề
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("📋 Đánh giá tất cả chủ đề", id="tab_all"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=3):
                        gr.Markdown(
                            f"Sẽ đánh giá tất cả **{len(topics)}** chủ đề trong `data/test/`: **{', '.join(topics) if topics else 'N/A'}**"
                        )
                    with gr.Column(scale=2):
                        chk_force_all = gr.Checkbox(
                            label="🔄 Buộc chạy lại Agent cho tất cả (Bỏ qua cache)",
                            value=False,
                        )
                    with gr.Column(scale=1):
                        btn_eval_all = gr.Button(
                            "🚀 Đánh giá tất cả",
                            variant="primary",
                            size="lg",
                        )

                status_all = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_all = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Đánh giá tất cả' để hiển thị tổng hợp</div>"
                        )
                    with gr.Column(scale=1):
                        chart_all = gr.Plot(label="Biểu đồ so sánh giữa các chủ đề")

                gr.Markdown("### 📊 Bảng tổng hợp điểm các chủ đề")
                summary_table_all = gr.Dataframe(
                    label="Tổng hợp điểm",
                    interactive=False,
                )

                gr.Markdown("### 📝 Bảng chi tiết toàn bộ câu hỏi")
                detail_table_all = gr.Dataframe(
                    label="Chi tiết toàn bộ câu hỏi",
                    interactive=False,
                    wrap=True,
                )

                btn_eval_all.click(
                    fn=evaluate_all_topics_handler,
                    inputs=[chk_force_all],
                    outputs=[
                        chart_all,
                        summary_table_all,
                        detail_table_all,
                        kpi_all,
                        status_all,
                    ],
                )

    return app


import socket


def find_available_port(start_port: int = 7860, max_search: int = 100) -> int:
    """Tìm port khả dụng để chạy Gradio app mà không bị lỗi đụng độ port."""
    for port in range(start_port, start_port + max_search):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    app = build_gradio_app()
    port = find_available_port(7860)
    logger.info(f"Khởi chạy Evaluation Dashboard tại: http://localhost:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
