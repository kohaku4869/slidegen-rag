import asyncio
import logging

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.graph.state import AgentState
from agent.tools.tools import SOFT_FAILURE_PREFIXES, TOOL_MAP

logger = logging.getLogger(__name__)


def _extract_chunks(raw: str) -> list[str]:
    """Tách raw tool output thành các chunk, bỏ chunk lỗi mềm (xem
    agent/tools/tools.py:SOFT_FAILURE_PREFIXES)."""
    if not raw or not isinstance(raw, str):
        return []
    chunks = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if chunk and not any(chunk.startswith(p) for p in SOFT_FAILURE_PREFIXES):
            chunks.append(chunk)
    return chunks


def _merge_documents(existing: list[str], new_chunks: list[str]) -> list[str]:
    """Merge và dedup documents qua các vòng retrieve."""
    seen = set(existing)
    merged = list(existing)
    for chunk in new_chunks:
        if chunk not in seen:
            seen.add(chunk)
            merged.append(chunk)
    return merged


async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Thực thi các tool_calls được tạo ra bởi retrieve_node.

    Đọc danh sách tool_calls từ state (không phải từ messages), thực thi
    song song các tool, sau đó merge kết quả mới vào documents hiện có.

    Luồng:
        retrieve_node → tool_node → grade_documents_node

    Args:
        state (AgentState): Cần có 'tool_calls'. 'documents' có thể có sẵn
                            từ các lần retrieve trước (sẽ được merge thêm).
        config (RunnableConfig): LangGraph tự truyền vào (annotation phải
                            đúng tên RunnableConfig để được inject). Được
                            forward vào tool_func.invoke để tool span nối
                            đúng vào trace của node này trên LangSmith và
                            TrajectoryCallbackHandler bắt được on_tool_*.

    Returns:
        dict: {"documents": list[str]} — tất cả raw chunks từ tool execution,
              merge với documents cũ (dedup).
    """
    tool_calls = state.get("tool_calls", [])
    existing_docs = state.get("documents", [])

    if not tool_calls:
        logger.warning("[tool_node] No tool_calls in state — skipping")
        return {"documents": existing_docs}

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
            logger.info(f"[tool_node] '{tool_name}' completed | output len={len(str(result))}")
            return str(result)
        except Exception as e:
            logger.error(f"[tool_node] '{tool_name}' failed: {e}")
            return ""

    raw_outputs = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])

    # Tách thành chunks và merge với existing documents
    new_chunks: list[str] = []
    for raw in raw_outputs:
        new_chunks.extend(_extract_chunks(raw))

    merged = _merge_documents(existing_docs, new_chunks)
    logger.info(
        f"[tool_node] +{len(new_chunks)} new chunks | total docs: {len(merged)}"
    )

    return {"documents": merged}
