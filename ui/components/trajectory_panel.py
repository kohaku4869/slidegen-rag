"""
components/trajectory_panel.py — Nội dung "Chi tiết chạy" hiện sau khi một
lượt agent xong: metric hàng ngang + bar chart tool usage + text đầy đủ
(xem plan6.md D1). Panel này nằm TRONG cùng `st.status` đã hiện bước trung
gian lúc đang chạy — không phải widget riêng (xem `ui/pages/chat.py`).
"""

from __future__ import annotations

import streamlit as st

from slidegen_rag.agent.observability.trajectory import Trajectory


def render_trajectory_summary(traj: Trajectory) -> None:
    usage = traj.token_usage or {}
    cols = st.columns(5)
    cols[0].metric("Thời gian", f"{traj.duration_s:.1f}s")
    cols[1].metric("Retrieve", f"{traj.retrieve_count}/{traj.max_retrieve}")
    cols[2].metric("Vòng lặp", traj.loop_count)
    cols[3].metric("Token Gemini", f"{usage.get('gemini_total_tokens', 0):,}")
    vnd = usage.get("total_cost_vnd", 0.0)
    cols[4].metric("Chi phí", f"{vnd:,.0f} đ" if vnd else "miễn phí (local)")

    if usage.get("local_calls"):
        st.caption(
            f"Local: miễn phí (self-hosted) — {usage['local_calls']} lượt, "
            f"{usage.get('local_total_tokens', 0):,} token"
        )

    tool_counts = traj.tool_usage_counts
    if tool_counts:
        st.bar_chart(dict(tool_counts))

    with st.expander("Trajectory đầy đủ"):
        st.code(traj.render_text(), language=None)
