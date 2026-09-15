"""
Ngân sách ký tự dùng chung cho context_evaluator_node và generate_node — cả
hai đọc cùng `documents` (append-only, oldest-first: chunk mới nhất là nội
dung vừa được thêm để đáp ứng missing_aspects của vòng trước), nên phải cắt
bớt theo cùng một quy tắc và cùng một ngưỡng để không suy luận lệch pha.
"""

CONTEXT_BUDGET_CHARS = 32_000


def build_context(
    documents: list[str],
    max_chars: int = CONTEXT_BUDGET_CHARS,
    priorities: list[int] | None = None,
) -> str:
    """
    Ghép `documents` thành 1 chuỗi context, cắt bớt khi vượt `max_chars`.

    Có 2 chế độ chọn chunk giữ lại khi vượt ngân sách:

    - `priorities=None` (mặc định, dùng khi không có điểm chấm — vd
      document_priorities thiếu/lệch kích thước): ưu tiên giữ NGUYÊN VẸN các
      chunk MỚI NHẤT (cuối list) — vì chunk mới là nội dung được thêm vào để
      đáp ứng missing_aspects của vòng trước; cắt từ đầu (như code cũ) sẽ
      vứt đúng phần đó. Duyệt từ CUỐI list về đầu, gom chunk cho tới khi
      thêm chunk tiếp theo sẽ vượt ngân sách.

    - `priorities` có giá trị (1:1 với `documents`, số CÀNG CAO càng ưu
      tiên giữ — xem grade_documents_node._PRIORITY_*): SẮP XẾP chunk theo
      (priority giảm dần, mới hơn trước) rồi gom theo kiểu tham lam — chunk
      'relevant' của vòng 1 được giữ trước chunk 'relevant_incomplete' của
      vòng 5, dù vòng 5 mới hơn. Lý do tồn tại: cắt theo thời gian thuần tuý
      vứt nhầm chunk CHẤT LƯỢNG CAO chỉ vì nó cũ (đo thực nghiệm trên
      pldc_v2 Q27: 57 chunk, 47% chunk giữ được đến từ vòng ≥3 nhưng keep-rate
      của grader rơi từ 76% ở vòng 1 xuống 32% ở vòng 3 — ngân sách 32K bị
      chunk vòng sau lấn chỗ chunk tốt của vòng đầu).

    Cả 2 chế độ: KHÔNG cắt giữa chunk (không mid-sentence); luôn giữ ít nhất
    1 chunk (nếu chunk được ưu tiên nhất một mình đã vượt max_chars, cắt
    CHÍNH chunk đó ở max_chars ký tự — last resort — chứ không trả về context
    rỗng); nếu có chunk bị bỏ, thêm ghi chú ở ĐẦU chuỗi kết quả.
    """
    if not documents:
        return ""

    joined = "\n\n".join(documents)
    if len(joined) <= max_chars:
        return joined

    use_priority = priorities is not None and len(priorities) == len(documents)

    if use_priority:
        # (priority giảm dần, index giảm dần) — cùng priority thì mới hơn
        # (index lớn hơn) được xét trước.
        order = sorted(range(len(documents)), key=lambda i: (-priorities[i], -i))
        drop_note = "lower-relevance chunk(s)"
    else:
        order = list(range(len(documents) - 1, -1, -1))  # mới nhất trước
        drop_note = "older chunk(s)"

    kept_idx: list[int] = []
    used = 0
    for i in order:
        add_len = len(documents[i]) + (2 if kept_idx else 0)  # "\n\n" separator
        if kept_idx and used + add_len > max_chars:
            continue
        kept_idx.append(i)
        used += add_len

    kept_idx.sort()  # khôi phục đúng thứ tự thời gian gốc để hiển thị
    kept = [documents[i] for i in kept_idx]

    dropped = len(documents) - len(kept)

    # Last resort: even the single most-favored chunk alone exceeds the budget.
    if len(kept) == 1 and len(kept[0]) > max_chars:
        kept[0] = kept[0][:max_chars]

    body = "\n\n".join(kept)
    if dropped:
        return f"[... {dropped} {drop_note} omitted for length ...]\n\n{body}"
    return body
