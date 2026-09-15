"""
Guard chống output sinh văn bản hỏng (degenerate) — dùng chung bởi
`agent/nodes/generate_node.py` (chặn TẠI NGUỒN, trước khi trả về cho người
dùng) và `evaluate/judge.py` (lớp phòng thủ thứ hai cho answer.txt cache
cũ đã sinh trước khi có guard này).

Bối cảnh: đã gặp 2 lần LLM rơi vào trạng thái lặp vô hạn — một câu trả lời
266,472 ký tự trong đó có chuỗi lặp 61,164 lần, và một câu 239,906 ký tự
toàn dấu gạch ngang. Trước đây guard chỉ có ở tầng evaluate nên nó cứu được
LLM judge khỏi bị ngập, nhưng KHÔNG cứu được điểm (câu trả lời thật sự hỏng
→ bị chấm 0 cả 4 tiêu chí) và cũng không bảo vệ gì cho người dùng thật.
"""

import re

# Ngưỡng 100 ký tự lặp liên tiếp: đủ cao để không đụng vào định dạng hợp lệ
# (đường kẻ ngang markdown "---" hay "===" dài nhất trong tài liệu cũng chỉ
# vài chục ký tự), đủ thấp để bắt được degenerate thật.
_REPEAT_RUN_RE = re.compile(r"(.)\1{99,}", re.DOTALL)


def collapse_pathological_repetition(text: str) -> str:
    """Rút gọn mỗi đoạn ký tự lặp bất thường thành một ghi chú ngắn.

    Trả về text đã làm sạch. So sánh với input để biết có degenerate hay
    không (xem `has_pathological_repetition`).
    """

    def _replace(match: "re.Match[str]") -> str:
        run = match.group(0)
        return f"{match.group(1)}... [lặp {len(run)} lần liên tiếp, đã rút gọn] "

    return _REPEAT_RUN_RE.sub(_replace, text)


def has_pathological_repetition(text: str) -> bool:
    """True nếu text chứa ít nhất một đoạn lặp bất thường."""
    return _REPEAT_RUN_RE.search(text) is not None
