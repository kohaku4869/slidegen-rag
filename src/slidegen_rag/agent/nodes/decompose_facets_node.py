import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.utils.prompts import (
    DECOMPOSE_FORCE_SPLIT_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    DECOMPOSE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# Cap cứng số facet fan-out song song — hằng số RIÊNG của module này (không
# đặt ở router.py) vì đây là fan-out cap (số nhánh Send), khác hẳn
# MAX_RETRIEVE (số vòng retry NỘI BỘ trong 1 facet).
MAX_FACETS = 4

# Marker so sánh — từ vựng ĐÓNG, không đếm số thực thể bằng regex (tiếng Việt
# không có cue viết hoa; đếm "và"/"với"/"hay" over-fire nặng, vd "điều kiện
# và hậu quả của giao dịch vô hiệu" là 1 chủ đề). Chỉ marker rõ ràng mới đủ
# tin cậy để kích hoạt hỏi lại.
_COMPARISON_MARKERS = (
    "so sánh",
    "phân biệt",
    "khác nhau",
    "khác gì",
    "đối chiếu",
    "so với",
    "compare",
    "difference between",
)

# Trigger hỏi lại chỉ khi câu hỏi đủ dài (câu quá ngắn có marker vẫn thường là
# 1 chủ đề, vd "phân biệt các loại quy phạm pháp luật").
_MIN_WORDS_FOR_FORCE_SPLIT = 8

# Bỏ bullet/đánh số ở đầu dòng ("- ", "* ", "• ", "1. ", "2) ") trước khi xét
# dòng có phải 1 sub-question hợp lệ hay không.
_BULLET_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s*")

# Dòng ngắn hơn ngưỡng này gần như chắc chắn là rác/heading trơ, không phải
# 1 sub-question thật.
_MIN_LINE_CHARS = 8

# temperature=0.0 theo house style _eval_llm/_grader_llm — parse text thủ
# công, không dùng structured output vì codebase hiện chưa dùng pattern đó ở
# đâu cả.
_decompose_llm = LLMModel(temperature=0.0, max_tokens=512)


def _make_label(i: int, sub_question: str) -> str:
    """Sinh label TRONG CODE thay vì tin LLM tự ghép LABEL với SUB-QUESTION —
    benchmark đo được 2/4 facet bị ghép lệch (nhãn "Dân sự" đi kèm sub-question
    "hình sự"), khiến facet_sufficiency mô tả sai khía cạnh cho generate_node.

    Tiền tố số thứ tự (bắt đầu từ 1) bảo đảm label KHÔNG BAO GIỜ trùng nhau
    dù 2 sub-question giống hệt nhau — trajectory.py dùng label làm key dict
    (ctx_by_facet, facet_retrieve_counts) và để map namespace_to_label theo
    đúng thứ tự Send được phát ra, nên key trùng sẽ làm mất dữ liệu 1 facet.
    Thay ","/"=" bằng khoảng trắng vì trajectory.py render dòng
    "Vòng lặp theo facet: {label}={n}, ..." — không làm vỡ evaluate (parser
    lấy nguyên đuôi dòng) nhưng ',' / '=' bên trong label sẽ khó đọc."""
    text = sub_question.replace(",", " ").replace("=", " ").strip()
    return f"{i + 1}. {text[:40]}…"


def _parse_facets(content: str, original_query: str) -> list[dict]:
    """
    Parse output: MỖI DÒNG LÀ 1 SUB-QUESTION độc lập — không còn LLM tự đặt
    LABEL nữa (xem _make_label). Dòng được coi là hợp lệ sau khi: bỏ
    khoảng trắng đầu/cuối, bỏ bullet/đánh số, bỏ nếu rỗng/là heading
    ("#"/"```")/là dòng dẫn nhập kết thúc bằng ":" (vd "Các khía cạnh:"), rồi
    (dung sai ngược cho LLM quen format cũ) nếu dòng còn ký tự "|" thì lấy
    phần sau dấu "|" CUỐI làm sub-question, và cuối cùng bỏ nếu quá ngắn.

    Enforce lại bằng code (không tin riêng LLM):
    - Parse lỗi/không dòng nào hợp lệ -> fail-closed về đúng 1 facet (câu hỏi
      gốc) -> KHÔNG BAO GIỜ trả 0 facet.
    - Đúng 1 dòng parse được (câu hỏi 1 chủ đề) -> sub-question BẮT BUỘC là
      nguyên văn câu hỏi gốc, bất kể LLM viết gì trong đó — đây là fast-path
      bắt buộc để không đổi hành vi/chi phí của câu hỏi đơn giản.
    - Cap cứng MAX_FACETS.
    """
    sub_questions: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _BULLET_RE.sub("", line).strip()
        if not line or line.startswith(("#", "```")) or line.endswith(":"):
            continue
        if "|" in line:
            line = line.rpartition("|")[2].strip()
        if len(line) < _MIN_LINE_CHARS:
            continue
        sub_questions.append(line)

    if not sub_questions or len(sub_questions) == 1:
        return [{"label": _make_label(0, original_query), "query": original_query}]

    sub_questions = sub_questions[:MAX_FACETS]
    return [
        {"label": _make_label(i, sq), "query": sq} for i, sq in enumerate(sub_questions)
    ]


def _should_force_split(facets: list[dict], query: str) -> bool:
    """Trigger khi CẢ 3 đúng: kết quả hiện tại chỉ có 1 facet, query chứa
    ≥1 marker so sánh, và query đủ dài — xem _COMPARISON_MARKERS."""
    if len(facets) != 1:
        return False
    if len(query.split()) < _MIN_WORDS_FOR_FORCE_SPLIT:
        return False
    q = query.lower()
    return any(marker in q for marker in _COMPARISON_MARKERS)


async def decompose_facets_node(state: dict) -> dict:
    """
    Tách câu hỏi gốc thành các facet độc lập, mỗi facet sẽ chạy song song qua
    facet_subgraph (xem agent/graph/builder.py::fan_out_facets).

    Luồng:
        START -> decompose_facets -> [fan-out Send] -> facet_subgraph (N bản)

    Args:
        state (AgentState): Cần có 'query'.

    Returns:
        dict: {"facet_queries": list[{"label": str, "query": str}]}
    """
    query = state["query"]
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=DECOMPOSE_USER_PROMPT.format(query=query)),
    ]

    try:
        response = await _decompose_llm.ainvoke(messages)
        facets = _parse_facets(response.content, query)

        if _should_force_split(facets, query):
            logger.info(
                f"[decompose_facets_node] 1 facet nhưng query có marker so sánh — "
                f"hỏi lại 1 lần: '{query[:80]}'"
            )
            try:
                retry_response = await _decompose_llm.ainvoke(
                    messages + [HumanMessage(content=DECOMPOSE_FORCE_SPLIT_PROMPT)]
                )
                retry_facets = _parse_facets(retry_response.content, query)
                if len(retry_facets) >= 2:
                    facets = retry_facets
                else:
                    logger.info(
                        "[decompose_facets_node] Hỏi lại vẫn 1 facet — giữ nguyên"
                    )
            except Exception as e:
                logger.warning(
                    f"[decompose_facets_node] Hỏi lại lỗi: {e} — giữ nguyên 1 facet"
                )
    except Exception as e:
        logger.error(
            f"[decompose_facets_node] LLM error: {e} — fallback về 1 facet (câu hỏi gốc)"
        )
        facets = [{"label": _make_label(0, query), "query": query}]

    logger.info(
        f"[decompose_facets_node] {len(facets)} facet(s): {[f['label'] for f in facets]}"
    )

    return {"facet_queries": facets}
