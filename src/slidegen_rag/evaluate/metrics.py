"""
Các chỉ số tất định (không cần LLM) đo độ phủ từ khóa giữa đáp án mẫu, chunk
đã truy hồi và câu trả lời của hệ thống.
"""

import re


def calculate_keyword_coverage(reference: str, used_chunks: list[str]) -> float:
    """Tính tỷ lệ bao phủ từ khóa cơ bản từ Reference xuất hiện trong các chunks.

    LƯU Ý (giữ nguyên để so sánh lịch sử với các lần chạy trước / README):
    đây là công thức GỐC, dùng SUBSTRING match (`w in all_chunks_text`) chứ
    không phải khớp theo ranh giới từ. Hệ quả: điểm số này tăng gần như đơn
    điệu theo SỐ LƯỢNG chunk, không phản ánh chất lượng — 40 chunk NGẪU
    NHIÊN lấy từ corpus đã đạt ~86.5% theo công thức này (đo thực nghiệm).
    Dùng `calculate_keyword_coverage_strict` để có con số đáng tin hơn; giữ
    cả hai để không làm mất khả năng đối chiếu với điểm cũ.
    """
    if not reference:
        return 100.0
    if not used_chunks:
        return 0.0

    all_chunks_text = " ".join(used_chunks).lower()
    words = re.findall(r"\b\w{3,}\b", reference.lower())
    if not words:
        return 100.0

    unique_words = list(set(words))
    matched = sum(1 for w in unique_words if w in all_chunks_text)
    coverage = (matched / len(unique_words)) * 100.0
    return min(100.0, coverage)


# Stopword tiếng Việt tối thiểu — chỉ các hư từ/liên từ cực phổ biến, không
# lọc từ có nghĩa chuyên ngành. Mục tiêu là loại các từ nối gần như luôn
# match dù chunk không liên quan (vd. "của", "là", "và"), không phải làm
# một bộ NLP stopword đầy đủ.
_VN_STOPWORDS = {
    "và",
    "của",
    "là",
    "có",
    "được",
    "cho",
    "các",
    "một",
    "những",
    "này",
    "đó",
    "khi",
    "với",
    "để",
    "trong",
    "về",
    "theo",
    "như",
    "từ",
    "hay",
    "hoặc",
    "thì",
    "nếu",
    "nên",
    "mà",
    "đã",
    "sẽ",
    "đang",
    "bị",
    "do",
    "vì",
    "nhưng",
    "còn",
    "cũng",
    "chỉ",
    "rất",
    "vào",
    "ra",
    "trên",
    "dưới",
    "giữa",
    "sau",
    "trước",
    "tại",
    "bởi",
    "cả",
    "mỗi",
    "tất",
}


def calculate_keyword_coverage_strict(reference: str, used_chunks: list[str]) -> float:
    """
    Bản coverage ĐÁNG TIN CẬY hơn: khớp theo RANH GIỚI TỪ (token-level, không
    phải substring), bỏ stopword tiếng Việt phổ biến, tối thiểu 2 ký tự/từ.

    So với `calculate_keyword_coverage`: substring match khiến số chunk càng
    nhiều thì điểm càng cao dù nội dung không liên quan (đo thực nghiệm: 40
    chunk ngẫu nhiên từ corpus đạt 86.5%). Token-set match không có hiệu ứng
    đó — thêm chunk không liên quan không làm tăng số từ khớp.
    """
    if not reference:
        return 100.0
    if not used_chunks:
        return 0.0

    chunk_tokens: set[str] = set()
    for c in used_chunks:
        chunk_tokens.update(re.findall(r"\b\w{2,}\b", c.lower()))

    ref_words = [
        w
        for w in re.findall(r"\b\w{2,}\b", reference.lower())
        if w not in _VN_STOPWORDS
    ]
    if not ref_words:
        return 100.0

    unique_words = set(ref_words)
    matched = sum(1 for w in unique_words if w in chunk_tokens)
    coverage = (matched / len(unique_words)) * 100.0
    return min(100.0, coverage)


def calculate_answer_coverage(reference: str, answer: str) -> float:
    """
    Như `calculate_keyword_coverage_strict` nhưng đo trên CÂU TRẢ LỜI CUỐI
    thay vì trên chunk đã retrieve — deterministic, KHÔNG dùng LLM.

    Lý do tồn tại: `coverage_pct_strict` chỉ nói "nguyên liệu có về tới
    context hay không", không nói agent có DÙNG được nó không. Đặt cạnh nhau,
    hai con số tách được 3 loại lỗi mà trước đây phải đoán qua feedback của
    judge (đo thực nghiệm trên run 20260914_003444):

        cov_chunk THẤP                      -> lỗi RETRIEVAL
            vd Q19: chunk 37.3% / answer 35.9% (dùng 96% — đã tận dụng hết
            nhưng nguyên liệu vốn đã thiếu)

        cov_chunk CAO, tỷ lệ dùng THẤP      -> lỗi GENERATION
            vd Q6: chunk 83.0% / answer 23.8% (dùng 29% — sinh văn bản hỏng)
            vd Q7: chunk 85.2% / answer 61.3% (dùng 72% — trả lời bị cắt)

        cov_chunk CAO, tỷ lệ dùng CAO, điểm judge THẤP -> lỗi JUDGE/REFERENCE
            vd Q28: chunk 72.1% / answer 70.9% (dùng 98%) mà vẫn 5.0 điểm —
            vì reference có lỗi dữ liệu, không phải agent làm sai

    Tương quan với total_score cao hơn hẳn coverage chunk (r=+0.55 so với
    +0.37), nhưng KHÔNG phải để thay điểm judge: câu trả lời đúng mà diễn đạt
    khác reference vẫn có thể coverage thấp (Q15 đạt 10.0 điểm với answer
    coverage chỉ 34.6%). Đây là công cụ CHẨN ĐOÁN chỉ ra tầng nào đang rò rỉ,
    không phải thước đo chất lượng.
    """
    return calculate_keyword_coverage_strict(reference, [answer] if answer else [])
