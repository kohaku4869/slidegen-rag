"""citations.py — Trích citation từ `final_state` của agent (thuần, không
Streamlit) — dùng chung bởi UI và REST API."""

from __future__ import annotations


def extract_citations(final_state: dict, max_chips: int = 8) -> list[dict]:
    """`zip(documents, document_node_ids, document_priorities)` là 1:1 (xem
    `agent/graph/state.py::AgentState`) — lọc `node_id is not None`, dedupe
    (giữ priority cao nhất nếu 1 node xuất hiện nhiều lần), sắp giảm dần theo
    priority, cắt `max_chips`."""
    documents = final_state.get("documents") or []
    node_ids = final_state.get("document_node_ids") or []
    priorities = final_state.get("document_priorities") or []

    seen: dict[int, dict] = {}
    for text, node_id, priority in zip(documents, node_ids, priorities):
        if node_id is None:
            continue
        existing = seen.get(node_id)
        if existing is None or priority > existing["priority"]:
            seen[node_id] = {"node_id": node_id, "text": text, "priority": priority}

    ordered = sorted(seen.values(), key=lambda c: c["priority"], reverse=True)
    return ordered[:max_chips]
