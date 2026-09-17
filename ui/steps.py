"""
steps.py — `render_step(step)`: dòng markdown ngắn cho MỘT `NodeStep` của
`TrajectoryRecorder` (xem plan6.md C4). Panel bước trung gian trực tiếp ở
`ui/pages/chat.py` bơm `on_step=queue.SimpleQueue().put` vào
`TrajectoryRecorder`, rồi `st.markdown(render_step(step))` mỗi lần rút được
một step khỏi hàng đợi — node-level, KHÔNG phải token-level (xem docstring
cuối file này và plan6.md C4).
"""

from __future__ import annotations

from slidegen_rag.agent.observability.trajectory import NodeStep


def render_step(step: NodeStep) -> str:
    """Một dòng markdown mô tả step đã hoàn tất. Không tự nhóm theo facet —
    `step.facet` chỉ được gắn tiền tố `[facet]`; việc nhóm dưới expander
    riêng cho từng facet là việc của trang gọi (fan-out các facet chạy song
    song nên các step interleave, không nên giả định thứ tự tuần tự)."""
    prefix = f"**[{step.facet}]** " if step.facet else ""
    header = f"{prefix}`{step.name}` — {step.duration_s:.2f}s"
    detail = _render_detail(step)
    return f"{header}  \n{detail}" if detail else header


def _render_detail(step: NodeStep) -> str:
    upd = step.update_summary

    if step.name == "retrieve":
        tool_calls = [tc for llm in step.llm_calls for tc in llm.tool_calls_emitted]
        if tool_calls:
            return "Tool calls: " + ", ".join(tool_calls)
        return ""

    if step.name == "grade_documents":
        # `update_summary` đã được `_summarize_update()` (trajectory.py) rút
        # list -> len(list) từ trước — "documents" ở đây LÀ số lượng (int),
        # không phải list, KHÔNG gọi len() lần nữa.
        n_documents = upd.get("documents")
        if n_documents is not None:
            return f"Tài liệu liên quan sau khi lọc: {n_documents}"
        return ""

    if step.name == "context_evaluator":
        sufficient = upd.get("context_sufficient")
        feedback = upd.get("feedback")
        badge = "✅ đủ ngữ cảnh" if sufficient else "❌ chưa đủ ngữ cảnh"
        return f"{badge}" + (f" — {feedback}" if feedback else "")

    if step.name == "generate":
        return "Đang soạn câu trả lời…"

    return ""
