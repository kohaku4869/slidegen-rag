import asyncio
import logging
import re

from langchain_core.runnables import RunnableConfig

from slidegen_rag.agent.graph.state import AgentState
from slidegen_rag.agent.tools.tools import (
    SOFT_FAILURE_PREFIXES,
    TOOL_MAP,
    describe_node,
)
from slidegen_rag.agent.utils.search_history import call_key, fingerprint

logger = logging.getLogger(__name__)

# Marker do agent/tools/tools.py gắn trước mỗi node (§N<id>|L<layer>§ ) để
# tool_node biết node nào ứng với đoạn text nào — phục vụ node_digest cho
# expand_node. Marker KHÔNG được lộ ra ngoài tool_node: không vào documents,
# không vào prompt sinh câu trả lời.
NODE_MARKER_RE = re.compile(r"^§N(-?\d+)\|L(-?\d+)§ ")

_MIN_CHUNK_CHARS = 20
_MAX_DIGEST_ENTRIES = 15
_DIGEST_SNIPPET_CHARS = 120
_DIGEST_HEADING_CHARS = 60


def _split_raw(raw: str, sep: str) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [p.strip() for p in raw.split(sep) if p.strip()]


def _extract_chunks(raw: str, tool_name: str) -> list[dict]:
    """
    Tách raw tool output thành các chunk. Mỗi chunk là 1 dict:
        {"text": str, "node_id": int|None, "layer": int|None}

    - Bỏ chunk lỗi mềm (xem agent/tools/tools.py:SOFT_FAILURE_PREFIXES).
    - Bỏ chunk quá ngắn (< 20 ký tự) — thường là rác/heading trơ.
    - web_search nối kết quả bằng "\\n\\n---\\n\\n" (khác semantic_search/
      keyword_search dùng "\\n\\n"), nên phải tách theo đúng separator của
      từng tool — nếu không, chunk "---" rác sẽ lọt vào và URL bị tách khỏi
      nội dung tương ứng.
    - Bóc marker §N<id>|L<layer>§ do semantic_search/keyword_search/
      expand_node gắn vào (nếu có) để phục vụ node_digest; chunk trả về
      KHÔNG còn marker.
    - Chunk từ web_search được gắn tiền tố "[WEB]" để generate_node và người
      đọc câu trả lời phân biệt được đâu là thông tin lấy từ internet, đâu là
      từ tài liệu gốc.
    """
    sep = "\n\n---\n\n" if tool_name == "web_search" else "\n\n"
    pieces = _split_raw(raw, sep)

    chunks = []
    for piece in pieces:
        if any(piece.startswith(p) for p in SOFT_FAILURE_PREFIXES):
            continue

        node_id, layer = None, None
        m = NODE_MARKER_RE.match(piece)
        if m:
            node_id, layer = int(m.group(1)), int(m.group(2))
            piece = piece[m.end() :].strip()

        if len(piece) < _MIN_CHUNK_CHARS:
            continue

        if tool_name == "web_search":
            piece = f"[WEB] {piece}"

        chunks.append({"text": piece, "node_id": node_id, "layer": layer})
    return chunks



def _format_heading(heading_path) -> str:
    """Ghép 2 thành phần cuối của heading_path (mục gần node nhất), cắt
    _DIGEST_HEADING_CHARS ký tự — đủ để LLM nhận ra node đang ở đâu trong
    cây mà không phình digest."""
    if not heading_path:
        return ""
    parts = [str(p) for p in list(heading_path)[-2:] if p]
    return " > ".join(parts)[:_DIGEST_HEADING_CHARS]


def _build_digest(chunk_infos: list[dict]) -> list[dict]:
    """
    Danh mục gọn các node vừa retrieve trong vòng này, để retrieve_node biết
    node nào có thể `expand_node` để lấy chi tiết thay vì search lại từ đầu.
    Ưu tiên node có khả năng có con lên đầu.

    Dùng `describe_node()` (metadata thật từ cây) thay vì đoán qua layer —
    cấp thêm n_children/n_siblings/has_parent/heading để retrieve_node biết
    NÊN chọn direction nào (children/siblings/parent) chứ không chỉ biết
    "có thể expand". `None` (không phải 0) đánh dấu "không tra được" (tree
    chưa sẵn sàng hoặc node_id lạ) — fallback về heuristic layer cũ chỉ khi
    đó, để _format_digest ở retrieve_node.py phân biệt được "0 con" thật với
    "không biết".
    """
    entries = []
    for info in chunk_infos:
        node_id = info["node_id"]
        if node_id is None:
            continue
        meta = describe_node(node_id)
        if meta is not None:
            has_children = meta.get("has_children", False)
            n_children = meta.get("n_children")
            n_siblings = meta.get("n_siblings")
            has_parent = meta.get("has_parent")
            heading = _format_heading(meta.get("heading_path"))
        else:
            has_children = (info["layer"] or 0) > 0
            n_children = None
            n_siblings = None
            has_parent = None
            heading = ""
        entries.append(
            {
                "id": node_id,
                "layer": info["layer"],
                "snippet": info["text"][:_DIGEST_SNIPPET_CHARS],
                "has_children": has_children,
                "n_children": n_children,
                "n_siblings": n_siblings,
                "has_parent": has_parent,
                "heading": heading,
            }
        )
    entries.sort(key=lambda e: not e["has_children"])

    seen_ids = set()
    unique = []
    for e in entries:
        if e["id"] in seen_ids:
            continue
        seen_ids.add(e["id"])
        unique.append(e)
    return unique[:_MAX_DIGEST_ENTRIES]


async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Thực thi các tool_calls được tạo ra bởi retrieve_node.

    Đọc danh sách tool_calls từ state (không phải từ messages), thực thi
    song song các tool, tách thành chunk và LỌC TRÙNG xuyên suốt mọi vòng
    bằng seen_fingerprints trước khi đưa vào pending_documents. Node này
    KHÔNG ghi "documents" — đó là việc của grade_documents_node (chỉ chấm
    chunk mới trong pending_documents rồi cộng phần đậu vào documents), để
    tránh chấm lại chunk đã từng đậu ở vòng trước.

    Luồng:
        retrieve_node → tool_node → grade_documents_node

    Args:
        state (AgentState): Cần có 'tool_calls'.
        config (RunnableConfig): LangGraph tự truyền vào (annotation phải
                            đúng tên RunnableConfig để được inject). Được
                            forward vào tool_func.invoke để tool span nối
                            đúng vào trace của node này trên LangSmith và
                            TrajectoryCallbackHandler bắt được on_tool_*.

    Returns:
        dict: {"pending_documents", "seen_fingerprints", "search_history",
               "node_digest"}.
    """
    tool_calls = state.get("tool_calls", [])
    seen = list(state.get("seen_fingerprints", []))
    seen_set = set(seen)
    history = list(state.get("search_history", []))
    round_no = state.get("retrieve_count", 0)

    if not tool_calls:
        logger.warning("[tool_node] No tool_calls in state — skipping")
        return {
            "pending_documents": [],
            "seen_fingerprints": seen,
            "search_history": history,
            "node_digest": [],
            "pending_meta": [],
        }

    logger.info(
        f"[tool_node] Executing {len(tool_calls)} tool(s): "
        f"{[tc['name'] for tc in tool_calls]}"
    )

    async def run_tool(tc: dict) -> str:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_func = TOOL_MAP.get(tool_name)

        if not tool_func:
            logger.error(f"[tool_node] Tool '{tool_name}' not found in TOOL_MAP")
            return ""

        try:
            result = await asyncio.to_thread(tool_func.invoke, tool_args, config)
            logger.info(
                f"[tool_node] '{tool_name}' completed | output len={len(str(result))}"
            )
            return str(result)
        except Exception as e:
            logger.error(f"[tool_node] '{tool_name}' failed: {e}")
            return ""

    raw_outputs = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])

    pending: list[str] = []
    pending_meta: list[dict] = []
    all_chunk_infos: list[dict] = []
    for tc, raw in zip(tool_calls, raw_outputs):
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {}) or {}
        chunk_infos = _extract_chunks(raw, tool_name)
        all_chunk_infos.extend(chunk_infos)

        fresh_count = 0
        for info in chunk_infos:
            fp = fingerprint(info["text"])
            if fp not in seen_set:
                seen_set.add(fp)
                seen.append(fp)
                pending.append(info["text"])
                pending_meta.append(
                    {"node_id": info["node_id"], "layer": info["layer"]}
                )
                fresh_count += 1

        soft_failed = bool(raw) and any(
            raw.startswith(p) for p in SOFT_FAILURE_PREFIXES
        )
        # expand_node (node_id+direction) và find_in_subtree (root_node_id+
        # patterns) không có "query" đơn — tái dùng đúng logic ghép khoá của
        # search_history.call_key thay vì nhân bản, để lịch sử ghi lại luôn
        # khớp với khoá dedup thật sự dùng ở retrieve_node. Các tool còn lại
        # giữ nguyên query gốc (không normalize) để hiển thị log dễ đọc hơn.
        if tool_name in ("expand_node", "find_in_subtree"):
            _, query_repr = call_key(tc)
        else:
            query_repr = tool_args.get("query", "")
        history.append(
            {
                "round": round_no,
                "tool": tool_name,
                "query": query_repr,
                "new_chunks": fresh_count,
                "soft_failed": soft_failed,
            }
        )

    node_digest = _build_digest(all_chunk_infos)

    logger.info(
        f"[tool_node] +{len(pending)} chunk mới (đã lọc trùng xuyên vòng) "
        f"| tổng đã thấy: {len(seen)}"
    )

    return {
        "pending_documents": pending,
        "seen_fingerprints": seen,
        "search_history": history,
        "node_digest": node_digest,
        "pending_meta": pending_meta,
    }
