"""pages/chat.py — Màn hình chat (xem plan6.md D1). Panel bước trung gian
nằm ĐÚNG chỗ panel trajectory — cùng một `st.status`, không nhảy layout."""

from __future__ import annotations

import queue
from uuid import UUID

import streamlit as st

from slidegen_rag.agent.graph.conversation import ConversationManager
from slidegen_rag.ui import data as ui_data
from slidegen_rag.ui.components.citations import (
    extract_citations,
    render_live_citations,
    render_replayed_citations,
)
from slidegen_rag.ui.components.trajectory_panel import render_trajectory_summary
from slidegen_rag.ui.steps import render_step

session_id_str = ui_data.resolve_id_param("active_session_id")
if not session_id_str:
    st.info("Chưa chọn phiên chat nào.")
    if st.button("← Về Projects"):
        st.switch_page("pages/projects.py")
    st.stop()

session_id = UUID(session_id_str)
session_repo = ui_data.get_session_repo()
document_repo = ui_data.get_document_repo()

state = ui_data.run_async(session_repo.load_session(session_id))
if state is None:
    st.error("Phiên chat không tồn tại (có thể đã bị xoá).")
    if st.button("← Về Projects"):
        st.switch_page("pages/projects.py")
    st.stop()

project_id_str = ui_data.resolve_id_param("active_project_id")
all_documents = ui_data.list_documents_cached(project_id_str) if project_id_str else []
ready_documents = [d for d in all_documents if d.status == "ready"]
current_scope = set(ui_data.run_async(session_repo.get_session_scope(session_id)))

st.title(f"💬 {state.title or f'Phiên {str(session_id)[:8]}'}")

with st.sidebar:
    if project_id_str and st.button("← Project"):
        st.switch_page("pages/project_home.py")
    st.subheader("Nguồn đang dùng")
    if not ready_documents:
        st.caption("Không có tài liệu sẵn sàng nào trong project.")
    selected_ids: list[UUID] = []
    for d in ready_documents:
        checked = st.checkbox(
            f"D{d.ref_no} · {d.title}", value=(d.id in current_scope), key=f"scope_{d.id}"
        )
        if checked:
            selected_ids.append(d.id)
    if set(selected_ids) != current_scope:
        ui_data.run_async(session_repo.set_session_scope(session_id, selected_ids))
        current_scope = set(selected_ids)
        st.rerun()

selected_documents = [d for d in ready_documents if d.id in current_scope]
if len(selected_documents) > 3:
    st.caption(f"{len(selected_documents)} tài liệu đang chọn — sẽ tiền lọc top-3 theo câu hỏi trước khi trả lời.")

merged_ra = ui_data.get_merged_ra(document_repo, selected_documents) if selected_documents else None
merged_tree = merged_ra.tree if merged_ra is not None else None

if merged_ra is None:
    st.warning("Chưa có tài liệu nào sẵn sàng trong phạm vi đang chọn — chọn ít nhất 1 tài liệu ở sidebar.")

# ── Replay lịch sử ───────────────────────────────────────────────────────
messages = ui_data.run_async(session_repo.list_messages(session_id))
assistant_ids = [m.id for m in messages if m.role == "assistant"]
citations_by_message = ui_data.run_async(session_repo.list_citations(assistant_ids)) if assistant_ids else {}

for m in messages:
    with st.chat_message(m.role):
        st.markdown(m.content)
        if m.role == "assistant":
            refs = citations_by_message.get(m.id, [])
            if refs and merged_tree is not None:
                render_replayed_citations(refs, merged_tree, document_repo, key_prefix=f"replay_{m.id}")
            meta = m.meta or {}
            if meta.get("trajectory_text"):
                summary = (
                    f"Chi tiết chạy · {meta.get('duration_s', 0):.1f}s · "
                    f"{meta.get('token_usage', {}).get('gemini_total_tokens', 0):,} token"
                )
                with st.expander(summary):
                    st.code(meta["trajectory_text"], language=None)

# ── Câu hỏi mới ──────────────────────────────────────────────────────────
query = st.chat_input(
    "Hỏi về tài liệu...", disabled=merged_ra is None
)
if query and merged_ra is not None:
    with st.chat_message("user"):
        st.markdown(query)

    conv = ui_data.run_async(
        ConversationManager.resume(session_repo, session_id, None, max_turns=state.max_turns)
    )
    standalone_query = ui_data.run_async(conv.contextualize(query))
    documents_for_turn = (
        ui_data.prefilter_documents_by_query(standalone_query, selected_documents)
        if len(selected_documents) > 3
        else selected_documents
    )
    turn_ra = (
        ui_data.get_merged_ra(document_repo, documents_for_turn)
        if documents_for_turn != selected_documents
        else merged_ra
    )

    runtime = ui_data.get_runtime()
    step_queue: queue.SimpleQueue = queue.SimpleQueue()
    future = runtime.submit(
        ui_data.run_agent_turn(
            runtime,
            turn_ra,
            query=standalone_query,
            web_search_enabled=True,
            conv=conv,
            on_step=step_queue.put,
            run_name=f"ui:{standalone_query[:60]}",
        )
    )

    with st.chat_message("assistant"):
        with st.status("Agent đang suy luận…", expanded=True) as status:
            while True:
                try:
                    step = step_queue.get(timeout=0.25)
                except queue.Empty:
                    if future.done():
                        break
                    continue
                st.markdown(render_step(step))

            try:
                result, traj = future.result()
            except Exception as e:
                status.update(label=f"Lỗi: {e}", state="error", expanded=True)
                st.error(f"Lượt hỏi thất bại: {e}")
                st.stop()

            answer = result.get("generation") or "Không có câu trả lời."
            usage = traj.token_usage or {}
            status.update(
                label=(
                    f"Xong · {traj.duration_s:.1f}s · "
                    f"{usage.get('gemini_total_tokens', 0):,} token"
                ),
                state="complete",
                expanded=False,
            )

        st.markdown(answer)

        citations = extract_citations(traj.final_state, max_chips=8)
        if citations and turn_ra is not None:
            render_live_citations(citations, turn_ra.tree, document_repo, key_prefix="live")

        with st.expander("Chi tiết chạy", expanded=False):
            render_trajectory_summary(traj)

    cited_refs = (
        ui_data.build_cited_refs(document_repo, turn_ra.tree, citations)
        if citations and turn_ra is not None
        else []
    )
    assistant_meta = {
        "duration_s": traj.duration_s,
        "retrieve_count": traj.retrieve_count,
        "loop_count": traj.loop_count,
        "tool_usage_counts": dict(traj.tool_usage_counts),
        "token_usage": usage,
        "run_id": traj.run_id,
        "langsmith_url": traj.langsmith_url,
        "trajectory_text": traj.render_text(),
    }
    document_node_ids = traj.final_state.get("document_node_ids", [])
    ui_data.run_async(
        conv.arecord_turn(
            query, standalone_query, answer, document_node_ids,
            cited_refs=cited_refs, assistant_meta=assistant_meta,
        )
    )
    ui_data.run_async(conv.maybe_compact())
    ui_data.invalidate_catalog()
    st.rerun()
