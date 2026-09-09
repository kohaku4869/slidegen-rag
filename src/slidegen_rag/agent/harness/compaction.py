"""
Context compaction — the `summarize` tool's actual effect on the message
list. A LangGraph `messages` channel annotated with `add_messages` can only
remove a whole message by id (`RemoveMessage`), never edit one in place, and
every prior tool call needs exactly one paired ToolMessage or the next LLM
call is malformed for both Gemini and OpenAI-style function calling. So
compaction removes BOTH sides of every previous tool-calling round together
(each AIMessage that carried tool_calls, and every ToolMessage) rather than
trying to selectively edit individual ToolMessage bodies — replacing them
with one plain HumanMessage that carries the model's own notes, a header
line for every reference ever seen (so it can still `open()` one it
discarded — see the paper's "retain candidate references" production
lesson), and the full node text for whichever references it chose to keep.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage

from slidegen_rag.agent.harness.doc_store import DocStore
from slidegen_rag.agent.harness.refs import ReferenceRegistry


def compact_messages(
    messages: list[AnyMessage],
    notes: str,
    keep_ref_ids: set[str],
    registry: ReferenceRegistry,
    store: DocStore,
) -> tuple[list[RemoveMessage], HumanMessage]:
    """Returns (removals, summary_message). Caller appends both (in either
    order — RemoveMessage only affects existing state, summary_message is a
    plain append) to the `messages` update for this graph step."""
    removals: list[RemoveMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            removals.append(RemoveMessage(id=m.id))
        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            removals.append(RemoveMessage(id=m.id))

    lines = [
        "[Ngữ cảnh đã được nén qua summarize — xem ghi chú và danh sách nguồn bên dưới]",
        f"Ghi chú của bạn ở vòng trước: {notes}",
        "",
        "Các nguồn đã thấy (id | tiêu đề | dòng nguồn):",
    ]
    for entry in registry.all_refs():
        lines.append(f"  [{entry.ref_id}] {entry.title or '(không tiêu đề)'} — dòng {entry.line_start}-{entry.line_end}")

    kept_entries = [e for e in registry.all_refs() if e.ref_id in keep_ref_ids]
    if kept_entries:
        lines.append("")
        lines.append("Nội dung đầy đủ được giữ lại cho các nguồn trên:")
        for entry in kept_entries:
            node = store.node(entry.node_id)
            lines.append(f"\n[{entry.ref_id}] {entry.title}\n{node.text}")

    lines.append("")
    lines.append(
        "Bạn có thể `open()` lại bất kỳ nguồn nào ở trên (kể cả nguồn không được giữ "
        "nội dung đầy đủ) nếu cần đọc lại."
    )

    return removals, HumanMessage(content="\n".join(lines))
