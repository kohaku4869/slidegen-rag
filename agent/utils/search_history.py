"""
Bộ nhớ truy vấn (query memory) dùng chung cho retrieve_node và
context_evaluator_node: theo dõi các cặp (tool, query) đã phát ra để
retrieve_node không lặp lại chúng, và context_evaluator nêu đúng cái đã thử
khi báo cáo feedback.
"""

import hashlib
import re


def normalize_query(q: str) -> str:
    """Chuẩn hoá query để so sánh: hạ chữ thường, gộp khoảng trắng."""
    return " ".join(str(q or "").lower().split())


def call_key(tc: dict) -> tuple[str, str]:
    """
    Khoá định danh một lần gọi tool = (tên tool, tham số định danh đã chuẩn hoá).

    Cố ý KHÔNG đưa top_k / search_depth vào khoá: hai lần gọi cùng tool cùng
    query nhưng khác top_k vẫn trả về context chồng lấn gần như hoàn toàn,
    nên vẫn coi là trùng để tránh lãng phí vòng retrieve.

    expand_node không có "query" mà có "node_id" — dùng node_id làm phần khoá
    thứ hai, nếu không mọi lần gọi expand_node đều rơi về cùng 1 khoá rỗng.
    """
    name = tc.get("name", "")
    args = tc.get("args", {}) or {}
    if name == "expand_node":
        return (name, str(args.get("node_id", "")))
    return (name, normalize_query(args.get("query", "")))


def tried_keys(history: list[dict]) -> set[tuple[str, str]]:
    """Tập hợp các (tool, query) đã từng phát ra, từ search_history trong state."""
    return {(h.get("tool", ""), normalize_query(h.get("query", ""))) for h in history}


def format_history(history: list[dict], limit: int = 12) -> str:
    """
    Render lịch sử tìm kiếm thành text ngắn gọn để đưa vào prompt, mới nhất
    trước. Nhấn mạnh các lần "0 chunk mới" — đó là tín hiệu mạnh nhất cho
    retrieve_node biết query nào đã cạn kiệt, không nên thử lại kiểu tương tự.
    """
    if not history:
        return ""
    lines = []
    for h in history[-limit:]:
        tool = h.get("tool", "?")
        query = h.get("query", "")
        n_new = h.get("new_chunks", 0)
        round_no = h.get("round", "?")
        note = " (0 chunk mới — coi như đã cạn)" if n_new == 0 else ""
        call_repr = f"{tool}(node_id={query})" if tool == "expand_node" else f"{tool}('{query}')"
        lines.append(f"- vòng {round_no} | {call_repr} → {n_new} chunk mới{note}")
    return "\n".join(lines)


_WORD_RE = re.compile(r"\S+")


def fingerprint(chunk: str) -> str:
    """Hash ngắn dùng để dedup chunk xuyên vòng (không phụ thuộc nội dung dài)."""
    normalized = " ".join(_WORD_RE.findall(chunk or ""))
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()
