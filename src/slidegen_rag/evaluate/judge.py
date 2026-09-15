"""
LLM-as-a-Judge: chấm câu trả lời trên 4 tiêu chí (accuracy, completeness,
relevance, faithfulness) theo thang điểm 10.

Mỗi tiêu chí được chấm bằng MỘT prompt độc lập (xem CRITERIA) thay vì gộp cả 4
vào một lượt — tránh "halo effect" khi LLM chấm cùng lúc. 4 lượt này chạy song
song nên không làm chậm việc chấm mỗi câu, chỉ tăng số lượt gọi LLM.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from slidegen_rag.evaluate.settings import (
    _MAX_EVAL_ANS_CHARS,
    _MAX_EVAL_CHUNKS_CHARS,
    _MAX_EVAL_REF_CHARS,
)

logger = logging.getLogger(__name__)

# Một số câu trả lời hiếm khi rơi vào trạng thái lặp ký tự/cụm từ vô hạn (LLM
# degenerate output) — hàng chục nghìn ký tự lặp lại làm judge hiểu nhầm là
# câu trả lời dài nhưng hợp lệ rồi trừ điểm oan vì bị cắt giữa chừng. Rút gọn
# các đoạn lặp bất thường (>= 100 ký tự lặp liên tiếp) TRƯỚC khi áp dụng cắt
# theo độ dài, để giới hạn ký tự phía trên không bị "ăn" hết vào rác lặp.
# Guard chống degenerate output giờ dùng chung với generate_node (nơi chặn TẠI
# NGUỒN). Giữ lớp này ở tầng judge làm phòng thủ thứ hai cho answer.txt cache cũ
# đã sinh ra trước khi generate_node có guard.
from slidegen_rag.agent.utils.text_guard import (  # noqa: E402
    collapse_pathological_repetition as _collapse_pathological_repetition,
)


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
                model="gemini-3.5-flash-lite",
                google_api_key=gemini_key,
                temperature=0,
                max_output_tokens=8192,
                timeout=120,
            )
            logger.info("[Judge] Fallback LLM: Gemini 2.5 Flash")
        except Exception as e:
            logger.warning(f"[Judge] Không thể tạo fallback Gemini LLM: {e}")

    return llm_primary, llm_fallback


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
