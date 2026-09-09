"""
evaluate_ragas.py — Giao diện Dashboard đánh giá chất lượng hệ thống RAG / CRAG Agent bằng RAGAS.
Đo lường 4 chỉ số cốt lõi chuẩn RAGAS và chuyển đổi toàn diện sang THANG ĐIỂM 10:
  1. Faithfulness (Tính trung thực / Chống ảo giác dựa trên Context Chunks)
  2. Answer Relevance (Độ liên quan và tập trung của câu trả lời so với câu hỏi)
  3. Context Recall (Độ thu hồi ngữ cảnh — mức độ tài liệu bao quát đáp án chuẩn)
  4. Context Precision (Độ chính xác ngữ cảnh — mức độ xếp hạng và độ liên quan của chunks)

Tự động cache câu trả lời vào answer.txt và các chunks tài liệu sử dụng vào chunks.txt
để tái sử dụng nhanh chóng mà không cần chạy lại Agent mỗi lần đánh giá.

Sử dụng:
    python evaluate/evaluate_ragas.py
"""

import asyncio
import json
import logging
import os
import re
import socket
import sys
import types
from contextlib import suppress
from pathlib import Path
from typing import Optional

# ── Fix compatibility cho Ragas & langchain_community & nest_asyncio trong Python 3.12 ──
for mod in [
    "langchain_community",
    "langchain_community.chat_models",
    "langchain_community.chat_models.vertexai",
    "langchain_community.llms",
    "langchain_community.llms.vertexai",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

class _DummyVertex:
    pass

if not hasattr(sys.modules["langchain_community.chat_models.vertexai"], "ChatVertexAI"):
    sys.modules["langchain_community.chat_models.vertexai"].ChatVertexAI = _DummyVertex
if not hasattr(sys.modules["langchain_community.llms"], "VertexAI"):
    sys.modules["langchain_community.llms"].VertexAI = _DummyVertex
if not hasattr(sys.modules["langchain_community.llms.vertexai"], "VertexAI"):
    sys.modules["langchain_community.llms.vertexai"].VertexAI = _DummyVertex

import nest_asyncio

def _safe_nest_run(main, *, debug=False, loop_factory=None, **kwargs):
    if loop_factory is not None:
        loop = loop_factory()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()
    loop.set_debug(debug)
    task = asyncio.ensure_future(main)
    try:
        return loop.run_until_complete(task)
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                loop.run_until_complete(task)

nest_asyncio.apply()
asyncio.run = _safe_nest_run

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

# ── Ragas imports ────────────────────────────────────────────────────────────
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ── Fix encoding console trên Windows ─────────────────────────────────────────
if sys.platform == "win32":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Đường dẫn & Cấu hình ─────────────────────────────────────────────────────
# repo root — 4 cấp trên: evaluate_ragas.py -> evaluate -> slidegen_rag -> src -> root
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
SCORE_GREEN = 8.5    # Xuất sắc
SCORE_AMBER = 7.0    # Đạt / Khá
COVERAGE_GREEN = 85.0
COVERAGE_AMBER = 70.0


# =============================================================================
# 1. DATA MODELS & SCHEMA (THANG ĐIỂM 10)
# =============================================================================

class RagasEvalResult(BaseModel):
    """Kết quả đánh giá Ragas cho 1 câu hỏi trên thang điểm 10."""
    faithfulness: Optional[float] = Field(
        default=None, description="Tính trung thực / Chống ảo giác (0.0 đến 10.0)."
    )
    answer_relevance: Optional[float] = Field(
        default=None, description="Độ liên quan câu trả lời (0.0 đến 10.0)."
    )
    context_recall: Optional[float] = Field(
        default=None, description="Độ thu hồi ngữ cảnh so với Reference (0.0 đến 10.0)."
    )
    context_precision: Optional[float] = Field(
        default=None, description="Độ chính xác và thứ hạng ngữ cảnh (0.0 đến 10.0)."
    )
    total_score: Optional[float] = Field(
        default=None, description="Tổng điểm trung bình của 4 tiêu chí Ragas (0.0 đến 10.0)."
    )
    feedback: str = Field(
        default="", description="Chi tiết phân tích chỉ số Ragas."
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
        if d.is_dir() and (d / "question.txt").exists() and (d / "reference.txt").exists():
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
            questions.append({
                "num": q_num,
                "text": match.group(2).strip(),
            })
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


# =============================================================================
# 3. KHỞI TẠO RAGAS EVALUATOR (LLM & EMBEDDINGS)
# =============================================================================

class CustomTreeEmbeddings(Embeddings):
    """Adapter tích hợp EmbeddingModel của slidegen-rag vào Langchain/Ragas."""
    def __init__(self):
        from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
        self.model = EmbeddingModel(
            cache_folder=ROOT_DIR / "models_cache",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.create_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.create_embedding(text)


def create_ragas_llm_and_embeddings():
    """Khởi tạo Ragas LLM Wrapper (Primary: OpenAI compatible, Fallback: Gemini) & Embeddings."""
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    ragas_llm_primary = None
    ragas_llm_fallback = None

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
            ragas_llm_primary = LangchainLLMWrapper(llm_primary)
            logger.info(f"[RAGAS Judge] Primary LLM: OpenAI-compatible @ {base_url}")
        except Exception as e:
            logger.warning(f"[RAGAS Judge] Không thể tạo primary LLM: {e}")

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
            ragas_llm_fallback = LangchainLLMWrapper(llm_fallback)
            logger.info("[RAGAS Judge] Fallback LLM: Gemini 2.5 Flash")
        except Exception as e:
            logger.warning(f"[RAGAS Judge] Không thể tạo fallback Gemini LLM: {e}")

    try:
        custom_emb = CustomTreeEmbeddings()
        ragas_embeddings = LangchainEmbeddingsWrapper(custom_emb)
        logger.info("[RAGAS Embeddings] Initialized CustomTreeEmbeddings successfully")
    except Exception as e:
        logger.error(f"[RAGAS Embeddings] Lỗi khởi tạo Embeddings: {e}")
        ragas_embeddings = None

    return ragas_llm_primary, ragas_llm_fallback, ragas_embeddings


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
    Nếu file answer.txt và chunks.txt đã tồn tại đầy đủ và force_rerun=False, đọc từ cache.
    """
    from slidegen_rag.main import load_ra
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.tools.tools import set_retrieval_augmentation, set_web_search_enabled
    from slidegen_rag.agent.observability.trajectory import make_initial_state

    question_file = DATA_TEST_DIR / topic / "question.txt"
    reference_file = DATA_TEST_DIR / topic / "reference.txt"
    answer_file = DATA_TEST_DIR / topic / "answer.txt"
    chunks_file = DATA_TEST_DIR / topic / "chunks.txt"
    index_path = DATA_INDEX_DIR / f"{topic}.pkl"

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
            progress_callback(f"[{topic}] Đã nạp {len(questions)} câu từ cache (answer.txt & chunks.txt)")
        logger.info(f"[{topic}] Sử dụng kết quả có sẵn từ answer.txt & chunks.txt")
        results = []
        for q in questions:
            num = q["num"]
            results.append({
                "num": num,
                "question": q["text"],
                "response": cached_answers.get(num, ""),
                "reference": references.get(num, ""),
                "used_chunks": cached_chunks.get(num, []),
            })
        return results, True

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy index: {index_path}")

    if progress_callback:
        progress_callback(f"[{topic}] Đang khởi tạo CRAG Agent...")

    ra = load_ra(index_path)
    set_retrieval_augmentation(ra)
    set_web_search_enabled(False)  # benchmark closed-book, xem evaluate.py
    graph = build_graph()

    results = []
    new_answers = {}
    new_chunks = {}
    total = len(questions)

    for idx, q in enumerate(questions):
        num = q["num"]
        q_text = q["text"]
        ref = references.get(num, "")

        if progress_callback:
            progress_callback(f"[{topic}] Đang chạy Agent sinh câu trả lời {num}/{total}...")

        logger.info(f"[{topic}] Agent đang trả lời câu {num}/{total}: {q_text[:70]}...")

        initial_state = make_initial_state(q_text)

        try:
            state_res = asyncio.run(graph.ainvoke(initial_state))
            answer = state_res.get("generation", "").strip()
            used_docs = state_res.get("documents", [])
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi chạy Agent câu {num}: {e}")
            answer = f"[LỖI AGENT] {e}"
            used_docs = []

        new_answers[num] = answer
        new_chunks[num] = used_docs

        results.append({
            "num": num,
            "question": q_text,
            "response": answer,
            "reference": ref,
            "used_chunks": used_docs,
        })

    save_answers(answer_file, new_answers)
    save_chunks(chunks_file, new_chunks)
    logger.info(f"[{topic}] Đã lưu câu trả lời vào {answer_file} và chunks vào {chunks_file}")

    return results, False


# =============================================================================
# 5. ĐÁNH GIÁ VỚI RAGAS METRICS (QUY ĐỔI SANG THANG ĐIỂM 10)
# =============================================================================

def calculate_keyword_coverage(reference: str, used_chunks: list[str]) -> float:
    """Tính tỷ lệ bao phủ từ khóa cơ bản từ Reference xuất hiện trong các chunks.

    LƯU Ý: dùng SUBSTRING match, không phải khớp theo ranh giới từ — điểm này
    tăng gần như đơn điệu theo SỐ LƯỢNG chunk (xem `calculate_keyword_coverage_strict`
    và evaluate/evaluate.py để có con số đáng tin hơn). Giữ hàm này để không
    mất khả năng đối chiếu với điểm cũ.
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


_VN_STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "các", "một", "những", "này",
    "đó", "khi", "với", "để", "trong", "về", "theo", "như", "từ", "hay",
    "hoặc", "thì", "nếu", "nên", "mà", "đã", "sẽ", "đang", "bị", "do",
    "vì", "nhưng", "còn", "cũng", "chỉ", "rất", "vào", "ra", "trên",
    "dưới", "giữa", "sau", "trước", "tại", "bởi", "cả", "mỗi", "tất",
}


def calculate_keyword_coverage_strict(reference: str, used_chunks: list[str]) -> float:
    """Bản coverage khớp theo ranh giới từ (token-set), bỏ stopword tiếng
    Việt — không bị thổi phồng bởi việc thêm chunk không liên quan. Xem
    evaluate/evaluate.py:calculate_keyword_coverage_strict để biết chi tiết."""
    if not reference:
        return 100.0
    if not used_chunks:
        return 0.0

    chunk_tokens: set[str] = set()
    for c in used_chunks:
        chunk_tokens.update(re.findall(r"\b\w{2,}\b", c.lower()))

    ref_words = [w for w in re.findall(r"\b\w{2,}\b", reference.lower()) if w not in _VN_STOPWORDS]
    if not ref_words:
        return 100.0

    unique_words = set(ref_words)
    matched = sum(1 for w in unique_words if w in chunk_tokens)
    coverage = (matched / len(unique_words)) * 100.0
    return min(100.0, coverage)


def _safe_scale_10(val: Optional[float]) -> Optional[float]:
    """Chuyển đổi điểm Ragas [0.0 - 1.0] sang Thang 10 [0.0 - 10.0]."""
    if val is None or np.isnan(val):
        return None
    return round(min(10.0, max(0.0, float(val) * 10.0)), 2)


async def _eval_one_metric_with_fallback(
    metric,
    sample: SingleTurnSample,
    primary_llm,
    fallback_llm,
    ragas_emb=None,
    metric_name: str = "",
) -> Optional[float]:
    """Đánh giá 1 metric với cơ chế Fallback riêng biệt: Primary -> Fallback Gemini."""
    # 1. Thử Primary LLM
    if primary_llm is not None:
        try:
            metric.llm = primary_llm
            if ragas_emb is not None and hasattr(metric, "embeddings"):
                metric.embeddings = ragas_emb
            score = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=90)
            if score is not None and not np.isnan(score):
                return float(score)
            logger.warning(f"[{metric_name}] Primary LLM trả về None/NaN, đang chuyển Fallback sang Gemini...")
        except Exception as e:
            logger.warning(f"[{metric_name}] Primary LLM thất bại ({e}), đang chuyển Fallback sang Gemini...")

    # 2. Fallback sang Gemini
    if fallback_llm is not None:
        try:
            metric.llm = fallback_llm
            if ragas_emb is not None and hasattr(metric, "embeddings"):
                metric.embeddings = ragas_emb
            score = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=90)
            if score is not None and not np.isnan(score):
                logger.info(f"[{metric_name}] Đánh giá thành công qua Fallback Gemini 2.5 Flash!")
                return float(score)
            logger.warning(f"[{metric_name}] Fallback Gemini trả về None/NaN.")
        except Exception as e:
            logger.error(f"[{metric_name}] Fallback Gemini thất bại: {e}")

    return None


async def _evaluate_all_metrics_with_fallback(
    primary_llm,
    fallback_llm,
    ragas_emb,
    sample: SingleTurnSample,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Đánh giá toàn bộ 4 chỉ số Ragas, từng chỉ số tự động fallback riêng rẽ sang Gemini."""
    f_raw = await _eval_one_metric_with_fallback(
        faithfulness, sample, primary_llm, fallback_llm, metric_name="Faithfulness"
    )
    ar_raw = await _eval_one_metric_with_fallback(
        answer_relevancy, sample, primary_llm, fallback_llm, ragas_emb=ragas_emb, metric_name="Answer Relevance"
    )
    cr_raw = await _eval_one_metric_with_fallback(
        context_recall, sample, primary_llm, fallback_llm, metric_name="Context Recall"
    )
    cp_raw = await _eval_one_metric_with_fallback(
        context_precision, sample, primary_llm, fallback_llm, metric_name="Context Precision"
    )
    return f_raw, ar_raw, cr_raw, cp_raw


def evaluate_single_answer_with_ragas(
    primary_llm,
    fallback_llm,
    ragas_emb,
    question: str,
    reference: str,
    response: str,
    used_chunks: list[str],
) -> Optional[RagasEvalResult]:
    """Đánh giá 1 câu hỏi bằng RAGAS với cơ chế Fallback per-metric và quy đổi Thang 10."""
    contexts = used_chunks if used_chunks else ["(Không có ngữ cảnh)"]
    ref_text = reference if reference else "(Không có đáp án mẫu)"
    ans_text = response if response else "(Không có câu trả lời)"

    sample = SingleTurnSample(
        user_input=question,
        response=ans_text,
        reference=ref_text,
        retrieved_contexts=contexts,
    )

    try:
        f_raw, ar_raw, cr_raw, cp_raw = asyncio.run(
            _evaluate_all_metrics_with_fallback(primary_llm, fallback_llm, ragas_emb, sample)
        )
    except Exception as e:
        logger.error(f"[Ragas Evaluator] Lỗi khi chạy đánh giá câu hỏi: {e}")
        f_raw, ar_raw, cr_raw, cp_raw = None, None, None, None

    # Chuyển đổi sang thang điểm 10
    f_10 = _safe_scale_10(f_raw)
    ar_10 = _safe_scale_10(ar_raw)
    cr_10 = _safe_scale_10(cr_raw)
    cp_10 = _safe_scale_10(cp_raw)

    valid_scores = [s for s in [f_10, ar_10, cr_10, cp_10] if s is not None]
    if not valid_scores:
        return None

    tot_10 = round(sum(valid_scores) / len(valid_scores), 2)

    fb_lines = []
    if f_10 is not None:
        fb_lines.append(f"• Faithfulness: {f_10}/10")
    if ar_10 is not None:
        fb_lines.append(f"• Answer Relevance: {ar_10}/10")
    if cr_10 is not None:
        fb_lines.append(f"• Context Recall: {cr_10}/10")
    if cp_10 is not None:
        fb_lines.append(f"• Context Precision: {cp_10}/10")

    feedback_text = "Đánh giá chuẩn RAGAS:\n" + "\n".join(fb_lines)

    return RagasEvalResult(
        faithfulness=f_10,
        answer_relevance=ar_10,
        context_recall=cr_10,
        context_precision=cp_10,
        total_score=tot_10,
        feedback=feedback_text,
    )


def run_evaluation_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict, bool]:
    """Thực hiện đánh giá toàn diện một chủ đề theo thang 10 bằng RAGAS."""
    results, was_cached = get_answers_and_chunks_for_topic(
        topic, force_rerun=force_rerun, progress_callback=progress_callback
    )

    primary_llm, fallback_llm, ragas_emb = create_ragas_llm_and_embeddings()
    if not primary_llm and not fallback_llm:
        raise RuntimeError("Không có LLM Judge nào khả dụng. Vui lòng kiểm tra LLM_BASE_URL hoặc GEMINI_API_KEY trong .env")

    rows = []
    total = len(results)

    for idx, item in enumerate(results, 1):
        num = item["num"]
        q_text = item["question"]
        resp = item["response"]
        ref = item["reference"]
        chunks = item["used_chunks"]

        if progress_callback:
            progress_callback(f"[{topic}] RAGAS đang chấm câu {idx}/{total}...")

        logger.info(f"[{topic}] RAGAS đang chấm câu {idx}/{total}...")

        eval_res = evaluate_single_answer_with_ragas(
            primary_llm, fallback_llm, ragas_emb, q_text, ref, resp, chunks
        )

        coverage = calculate_keyword_coverage(ref, chunks)
        coverage_strict = calculate_keyword_coverage_strict(ref, chunks)

        if eval_res is not None:
            is_success = True
            faith_val = eval_res.faithfulness
            ans_rel_val = eval_res.answer_relevance
            ctx_rec_val = eval_res.context_recall
            ctx_prec_val = eval_res.context_precision
            total_score = eval_res.total_score
            feedback_val = eval_res.feedback
        else:
            is_success = False
            faith_val = None
            ans_rel_val = None
            ctx_rec_val = None
            ctx_prec_val = None
            total_score = None
            feedback_val = "[LỖI ĐÁNH GIÁ] Không nhận được phản hồi hợp lệ từ Ragas Evaluator."

        rows.append({
            "STT": num,
            "topic": topic,
            "question": q_text,
            "response": resp,
            "reference": ref,
            "faithfulness": faith_val,
            "answer_relevance": ans_rel_val,
            "context_recall": ctx_rec_val,
            "context_precision": ctx_prec_val,
            "total_score": total_score,
            "coverage_pct": round(coverage, 1),
            "coverage_pct_strict": round(coverage_strict, 1),
            "chunks_count": len(chunks),
            "feedback": feedback_val,
            "is_success": is_success,
            "used_chunks": chunks,
        })

    detail_df = pd.DataFrame(rows)

    valid_df = detail_df[detail_df["is_success"] == True]
    success_count = len(valid_df)
    failed_count = len(detail_df) - success_count

    if not valid_df.empty:
        summary = {
            "faithfulness": float(valid_df["faithfulness"].dropna().astype(float).mean()) if not valid_df["faithfulness"].dropna().empty else 0.0,
            "answer_relevance": float(valid_df["answer_relevance"].dropna().astype(float).mean()) if not valid_df["answer_relevance"].dropna().empty else 0.0,
            "context_recall": float(valid_df["context_recall"].dropna().astype(float).mean()) if not valid_df["context_recall"].dropna().empty else 0.0,
            "context_precision": float(valid_df["context_precision"].dropna().astype(float).mean()) if not valid_df["context_precision"].dropna().empty else 0.0,
            "total_score": float(valid_df["total_score"].dropna().astype(float).mean()) if not valid_df["total_score"].dropna().empty else 0.0,
            "coverage_pct": float(valid_df["coverage_pct"].astype(float).mean()),
            "coverage_pct_strict": float(valid_df["coverage_pct_strict"].astype(float).mean()),
            "avg_chunks": float(detail_df["chunks_count"].mean()),
            "total_questions": len(detail_df),
            "success_count": success_count,
            "failed_count": failed_count,
        }
    else:
        summary = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
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
# 6. HTML KPI CARDS & VISUALIZATION (THANG ĐIỂM 10)
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


def generate_summary_kpi_html(summary: dict, topic_name: str = "", was_cached: bool = False) -> str:
    """Tạo toàn bộ khối KPI metrics HTML cho RAGAS Dashboard."""
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
            <span style="font-size: 15px; font-weight: 700; color: #1e293b;">📊 Tổng quan đánh giá RAGAS (Thang 10): {topic_name}</span>
            {cache_badge}
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
            {format_metric_html("🏆 Tổng điểm RAGAS", summary.get("total_score"))}
            {format_metric_html("🛡️ Trung thực (Faithfulness)", summary.get("faithfulness"))}
            {format_metric_html("🎯 Độ liên quan (Answer Relevance)", summary.get("answer_relevance"))}
            {format_metric_html("📚 Thu hồi ngữ cảnh (Context Recall)", summary.get("context_recall"))}
            {format_metric_html("🎯 Chính xác ngữ cảnh (Context Precision)", summary.get("context_precision"))}
            {format_metric_html("🔍 Phủ từ khóa (Keyword Coverage)", summary.get("coverage_pct"), is_percentage=True)}
        </div>
        <div style="margin-top: 10px; padding: 8px 12px; background-color: #f1f5f9; border-radius: 6px; font-size: 12px; color: #475569; text-align: center;">
            Tổng số câu hỏi: <b>{summary.get('total_questions', 0)}</b> (Thành công: <b>{summary.get('success_count', 0)}</b>{failed_badge}) | Trung bình Context: <b>{summary.get('avg_chunks', 0.0):.1f} chunks/câu</b>
        </div>
    </div>
    """


def create_single_topic_chart(summary: dict, topic: str) -> plt.Figure:
    """Tạo biểu đồ cột thể hiện 4 chỉ số Ragas của một chủ đề trên Thang 10."""
    metrics = ["Faithfulness", "Answer Relevance", "Context Recall", "Context Precision", "Total Score"]
    scores = [
        summary.get("faithfulness", 0.0),
        summary.get("answer_relevance", 0.0),
        summary.get("context_recall", 0.0),
        summary.get("context_precision", 0.0),
        summary.get("total_score", 0.0),
    ]

    colors = [get_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=100)
    bars = ax.bar(metrics, scores, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            color="#1e293b",
        )

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm RAGAS (Thang 10)", fontsize=11, fontweight="600")
    ax.set_title(f"Phân tích RAGAS Metrics — Chủ đề: {topic}", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(y=SCORE_GREEN, color="#10b981", linestyle="--", alpha=0.6, label="Mục tiêu (>= 8.5)")
    ax.axhline(y=SCORE_AMBER, color="#f59e0b", linestyle=":", alpha=0.6, label="Ngưỡng đạt (>= 7.0)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_all_topics_chart(all_summaries: dict[str, dict]) -> plt.Figure:
    """Tạo biểu đồ so sánh 4 chỉ số Ragas giữa các chủ đề trên Thang 10."""
    topics = list(all_summaries.keys())
    if not topics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        return fig

    x = np.arange(len(topics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(topics) * 2.5), 5), dpi=100)

    faith = [all_summaries[t].get("faithfulness", 0.0) for t in topics]
    ans_rel = [all_summaries[t].get("answer_relevance", 0.0) for t in topics]
    ctx_rec = [all_summaries[t].get("context_recall", 0.0) for t in topics]
    ctx_prec = [all_summaries[t].get("context_precision", 0.0) for t in topics]

    ax.bar(x - 1.5 * width, faith, width, label="Faithfulness", color="#10b981", edgecolor="white")
    ax.bar(x - 0.5 * width, ans_rel, width, label="Answer Relevance", color="#3b82f6", edgecolor="white")
    ax.bar(x + 0.5 * width, ctx_rec, width, label="Context Recall", color="#8b5cf6", edgecolor="white")
    ax.bar(x + 1.5 * width, ctx_prec, width, label="Context Precision", color="#06b6d4", edgecolor="white")

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm RAGAS (Thang 10)", fontsize=11, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, fontsize=11, fontweight="600")
    ax.set_title("So sánh các chỉ số RAGAS giữa các chủ đề (Thang 10)", fontsize=14, fontweight="bold", pad=12)
    ax.axhline(y=SCORE_GREEN, color="#10b981", linestyle="--", alpha=0.5, label="Mục tiêu (8.5)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_summary_table(all_summaries: dict[str, dict]) -> pd.DataFrame:
    """Tạo bảng tổng hợp điểm RAGAS cho các chủ đề."""
    rows = []
    for topic, summary in all_summaries.items():
        rows.append({
            "Chủ đề (Topic)": topic,
            "Faithfulness": round(summary.get("faithfulness", 0.0), 2),
            "Answer Relevance": round(summary.get("answer_relevance", 0.0), 2),
            "Context Recall": round(summary.get("context_recall", 0.0), 2),
            "Context Precision": round(summary.get("context_precision", 0.0), 2),
            "Total Score": round(summary.get("total_score", 0.0), 2),
            "Keyword Coverage (%)": f"{summary.get('coverage_pct', 0.0):.1f}%",
            "Avg Chunks": round(summary.get("avg_chunks", 0.0), 1),
            "Số câu thành công": f"{summary.get('success_count', 0)}/{summary.get('total_questions', 0)}",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. GRADIO APP & HANDLERS
# =============================================================================

_current_detail_df: Optional[pd.DataFrame] = None


def evaluate_single_topic_handler(topic: str, force_rerun: bool, progress=gr.Progress()):
    """Handler đánh giá một chủ đề bằng RAGAS."""
    global _current_detail_df

    if not topic:
        return None, None, "<p>⚠️ Vui lòng chọn một chủ đề!</p>", gr.Dropdown(choices=[]), "⚠️ Vui lòng chọn một chủ đề!"

    try:
        progress(0.1, desc=f"Đang chuẩn bị dữ liệu cho chủ đề: {topic}...")

        def cb(msg):
            progress(0.4, desc=msg)

        detail_df, summary, was_cached = run_evaluation_for_topic(
            topic, force_rerun=force_rerun, progress_callback=cb
        )

        _current_detail_df = detail_df

        progress(0.85, desc="Đang tạo biểu đồ & báo cáo RAGAS...")

        kpi_html = generate_summary_kpi_html(summary, topic_name=topic, was_cached=was_cached)
        chart = create_single_topic_chart(summary, topic)

        display_cols = [
            "STT", "question", "response",
            "faithfulness", "answer_relevance", "context_recall", "context_precision",
            "total_score", "coverage_pct", "chunks_count", "feedback"
        ]
        display_df = detail_df[display_cols].copy()
        display_df = display_df.fillna("-")

        q_choices = [f"Câu {row['STT']}: {str(row['question'])[:60]}..." for _, row in detail_df.iterrows()]
        first_choice = q_choices[0] if q_choices else None

        succ = summary.get("success_count", 0)
        tot = summary.get("total_questions", 0)
        fail = summary.get("failed_count", 0)
        fail_msg = f" (⚠️ {fail} câu lỗi không tính vào điểm tổng)" if fail > 0 else ""
        status_msg = f"✅ Hoàn tất đánh giá RAGAS chủ đề **{topic}** (Thành công {succ}/{tot} câu{fail_msg})!"

        progress(1.0, desc="Hoàn tất!")

        return chart, display_df, kpi_html, gr.Dropdown(choices=q_choices, value=first_choice), status_msg

    except Exception as e:
        logger.error(f"Lỗi đánh giá {topic}: {e}", exc_info=True)
        err_msg = f"❌ Lỗi: {e}\n\n{traceback.format_exc()}"
        return None, None, f"<div style='color: red;'>{err_msg}</div>", gr.Dropdown(choices=[]), err_msg


def inspect_question_handler(selected_q_text: str):
    """Handler hiển thị chi tiết câu hỏi, đáp án, chunks và Ragas metrics."""
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

    col1_content = f"""### ❓ Câu hỏi #{item['STT']}
**{item['question']}**

---
### 📖 Đáp án mẫu (Reference Answer)
{item['reference'] if item['reference'] else '*(Không có đáp án mẫu)*'}

---
### 🤖 Câu trả lời của CRAG Agent
{item['response']}
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

    faith_str = f"{item['faithfulness']}/10" if pd.notna(item['faithfulness']) else "N/A"
    ar_str = f"{item['answer_relevance']}/10" if pd.notna(item['answer_relevance']) else "N/A"
    cr_str = f"{item['context_recall']}/10" if pd.notna(item['context_recall']) else "N/A"
    cp_str = f"{item['context_precision']}/10" if pd.notna(item['context_precision']) else "N/A"
    tot_str = f"{item['total_score']}/10" if pd.notna(item['total_score']) else "N/A"

    col3_content = f"""### ⚖️ Đánh giá từ RAGAS Metrics (Thang 10)
- **Tính trung thực (Faithfulness)**: `{faith_str}`
- **Độ liên quan (Answer Relevance)**: `{ar_str}`
- **Thu hồi ngữ cảnh (Context Recall)**: `{cr_str}`
- **Chính xác ngữ cảnh (Context Precision)**: `{cp_str}`
- **🏆 Tổng điểm (Total Score)**: `{tot_str}`
- **Độ phủ từ khóa (Keyword Coverage)**: `{item['coverage_pct']}%`

---
### 📝 Chi tiết RAGAS
```
{item['feedback']}
```
"""

    return col1_content, col2_content, col3_content


def evaluate_all_topics_handler(force_rerun: bool, progress=gr.Progress()):
    """Handler đánh giá toàn bộ các chủ đề bằng RAGAS."""
    topics = get_available_topics()
    if not topics:
        return None, None, None, "<p>⚠️ Không tìm thấy chủ đề nào trong data/test/</p>", "⚠️ Không có dữ liệu."

    all_summaries = {}
    all_details = []
    total_questions = 0
    total_success = 0
    total_failed = 0

    for t_idx, topic in enumerate(topics):
        try:
            base_prog = t_idx / len(topics)
            progress(base_prog, desc=f"[{t_idx+1}/{len(topics)}] Đang xử lý: {topic}...")

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

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá chủ đề {topic}: {e}", exc_info=True)
            all_summaries[topic] = {
                "faithfulness": 0.0, "answer_relevance": 0.0, "context_recall": 0.0,
                "context_precision": 0.0, "total_score": 0.0, "coverage_pct": 0.0,
                "avg_chunks": 0.0, "total_questions": 0, "success_count": 0,
                "failed_count": 0, "error": str(e)
            }

    progress(0.9, desc="Đang tạo biểu đồ tổng hợp RAGAS...")

    chart = create_all_topics_chart(all_summaries)
    summary_df = create_summary_table(all_summaries)

    if all_details:
        combined_df = pd.concat(all_details, ignore_index=True)
        display_cols = [
            "STT", "topic", "question", "response",
            "faithfulness", "answer_relevance", "context_recall", "context_precision",
            "total_score", "coverage_pct", "chunks_count", "feedback"
        ]
        detail_table_df = combined_df[[c for c in display_cols if c in combined_df.columns]].copy()
        detail_table_df = detail_table_df.fillna("-")
    else:
        detail_table_df = pd.DataFrame()

    valid_summaries = [s for s in all_summaries.values() if "error" not in s and s.get("success_count", 0) > 0]
    if valid_summaries:
        overall_summary = {
            "total_score": sum(s["total_score"] for s in valid_summaries) / len(valid_summaries),
            "faithfulness": sum(s["faithfulness"] for s in valid_summaries) / len(valid_summaries),
            "answer_relevance": sum(s["answer_relevance"] for s in valid_summaries) / len(valid_summaries),
            "context_recall": sum(s["context_recall"] for s in valid_summaries) / len(valid_summaries),
            "context_precision": sum(s["context_precision"] for s in valid_summaries) / len(valid_summaries),
            "coverage_pct": sum(s["coverage_pct"] for s in valid_summaries) / len(valid_summaries),
            "avg_chunks": sum(s["avg_chunks"] for s in valid_summaries) / len(valid_summaries),
            "total_questions": total_questions,
            "success_count": total_success,
            "failed_count": total_failed,
        }
    else:
        overall_summary = {}

    kpi_html = generate_summary_kpi_html(overall_summary, topic_name="Toàn hệ thống (All Topics)", was_cached=not force_rerun)

    fail_msg = f" (⚠️ {total_failed} câu lỗi không tính vào điểm tổng)" if total_failed > 0 else ""
    status_msg = f"✅ Đã hoàn tất đánh giá {len(valid_summaries)}/{len(topics)} chủ đề (Thành công {total_success}/{total_questions} câu{fail_msg})!"

    progress(1.0, desc="Hoàn tất!")

    return chart, summary_df, detail_table_df, kpi_html, status_msg


# =============================================================================
# 8. GIAO DIỆN GRADIO (THEME & BLOCKS)
# =============================================================================

def build_gradio_app():
    """Khởi tạo giao diện Dashboard Gradio cho RAGAS."""
    topics = get_available_topics()
    topic_choices = topics if topics else ["(Không có chủ đề)"]

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        font=["Inter", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="SLIDEGEN-RAG — Ragas Evaluation Dashboard") as app:
        gr.Markdown(
            """
            # 📊 SLIDEGEN-RAG — Ragas Evaluation Dashboard
            Hệ thống đánh giá chuyên sâu **CRAG Agent** sử dụng chuẩn framework **RAGAS** (Đã chuyển đổi sang **Thang điểm 10**).
            
            **4 Chỉ số RAGAS cốt lõi:**
            - 🛡️ **Faithfulness (Tính trung thực / Chống ảo giác)**: Đo lường mức độ câu trả lời dựa trên context chunks đã trích xuất.
            - 🎯 **Answer Relevance (Độ liên quan câu trả lời)**: Đo lường mức độ câu trả lời giải quyết trực tiếp câu hỏi.
            - 📚 **Context Recall (Độ thu hồi ngữ cảnh)**: Đo lường mức độ context chunks bao quát câu trả lời chuẩn (Reference).
            - 🎯 **Context Precision (Độ chính xác ngữ cảnh)**: Đo lường chất lượng xếp hạng và mức độ liên quan của các chunks được truy xuất.
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
                            "🚀 Bắt đầu đánh giá RAGAS",
                            variant="primary",
                            size="lg",
                        )

                status_single = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_single = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Bắt đầu đánh giá RAGAS' để hiển thị các chỉ số</div>"
                        )
                    with gr.Column(scale=1):
                        chart_single = gr.Plot(label="Biểu đồ phân tích RAGAS Metrics")

                gr.Markdown("### 📋 Bảng điểm chi tiết từng câu hỏi (Thang điểm 10)")
                table_single = gr.Dataframe(
                    label="Chi tiết kết quả đánh giá RAGAS",
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
                        inspect_col1 = gr.Markdown("*(Chọn một câu hỏi ở trên để xem chi tiết)*")
                    with gr.Column(scale=1):
                        inspect_col2 = gr.Markdown("")
                    with gr.Column(scale=1):
                        inspect_col3 = gr.Markdown("")

                btn_eval_single.click(
                    fn=evaluate_single_topic_handler,
                    inputs=[topic_dropdown, chk_force_single],
                    outputs=[chart_single, table_single, kpi_single, inspect_dropdown, status_single],
                )

                inspect_dropdown.change(
                    fn=inspect_question_handler,
                    inputs=[inspect_dropdown],
                    outputs=[inspect_col1, inspect_col2, inspect_col3],
                )

            # ─────────────────────────────────────────────────────────────────
            # TAB 2: Đánh giá tất cả chủ đề
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("📋 Đánh giá tất cả chủ đề", id="tab_all"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=3):
                        gr.Markdown(f"Sẽ đánh giá tất cả **{len(topics)}** chủ đề trong `data/test/`: **{', '.join(topics) if topics else 'N/A'}**")
                    with gr.Column(scale=2):
                        chk_force_all = gr.Checkbox(
                            label="🔄 Buộc chạy lại Agent cho tất cả (Bỏ qua cache)",
                            value=False,
                        )
                    with gr.Column(scale=1):
                        btn_eval_all = gr.Button(
                            "🚀 Đánh giá tất cả bằng RAGAS",
                            variant="primary",
                            size="lg",
                        )

                status_all = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_all = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Đánh giá tất cả bằng RAGAS' để hiển thị tổng hợp</div>"
                        )
                    with gr.Column(scale=1):
                        chart_all = gr.Plot(label="Biểu đồ so sánh RAGAS giữa các chủ đề")

                gr.Markdown("### 📊 Bảng tổng hợp điểm RAGAS các chủ đề")
                summary_table_all = gr.Dataframe(
                    label="Tổng hợp điểm RAGAS",
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
                    outputs=[chart_all, summary_table_all, detail_table_all, kpi_all, status_all],
                )

    return app


def find_available_port(start_port: int = 7861, max_search: int = 100) -> int:
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
    port = find_available_port(7861)
    logger.info(f"Khởi chạy RAGAS Evaluation Dashboard tại: http://localhost:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

