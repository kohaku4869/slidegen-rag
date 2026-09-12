"""
Ngân sách ký tự dùng chung cho context_evaluator_node và generate_node — cả
hai đọc cùng `documents` (append-only, oldest-first: chunk mới nhất là nội
dung vừa được thêm để đáp ứng missing_aspects của vòng trước), nên phải cắt
bớt theo cùng một quy tắc và cùng một ngưỡng để không suy luận lệch pha.
"""

CONTEXT_BUDGET_CHARS = 32_000


def build_context(documents: list[str], max_chars: int = CONTEXT_BUDGET_CHARS) -> str:
    """
    Ghép `documents` thành 1 chuỗi context, ưu tiên giữ NGUYÊN VẸN các chunk
    MỚI NHẤT (cuối list) khi vượt ngân sách — vì chunk mới là nội dung được
    thêm vào để đáp ứng missing_aspects của vòng trước; cắt từ đầu (như code
    cũ) sẽ vứt đúng phần đó.

    - Nếu tổng độ dài (kể cả separator) <= max_chars: join "\\n\\n" như cũ.
    - Nếu vượt: duyệt từ CUỐI list về đầu, gom chunk cho tới khi thêm chunk
      tiếp theo sẽ vượt ngân sách, rồi đảo lại đúng thứ tự thời gian gốc.
      KHÔNG cắt giữa chunk (không mid-sentence).
    - Luôn giữ ít nhất 1 chunk: nếu chunk cuối cùng (mới nhất) một mình đã
      vượt max_chars, cắt CHÍNH chunk đó ở max_chars ký tự (last resort) chứ
      không trả về context rỗng.
    - Nếu có chunk bị bỏ, thêm ghi chú
      "[... N older chunk(s) omitted for length ...]" ở ĐẦU chuỗi kết quả
      (trước các chunk được giữ) để rõ ràng đây là các chunk CŨ hơn bị bỏ,
      không phải bị cắt giữa chừng.
    """
    if not documents:
        return ""

    joined = "\n\n".join(documents)
    if len(joined) <= max_chars:
        return joined

    kept: list[str] = []
    used = 0
    for doc in reversed(documents):
        add_len = len(doc) + (2 if kept else 0)  # "\n\n" separator
        if kept and used + add_len > max_chars:
            break
        kept.append(doc)
        used += add_len
    kept.reverse()

    dropped = len(documents) - len(kept)

    # Last resort: even the single newest chunk alone exceeds the budget.
    if len(kept) == 1 and len(kept[0]) > max_chars:
        kept[0] = kept[0][:max_chars]

    body = "\n\n".join(kept)
    if dropped:
        return f"[... {dropped} older chunk(s) omitted for length ...]\n\n{body}"
    return body
