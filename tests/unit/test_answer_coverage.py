"""
Unit tests cho metric `calculate_answer_coverage` — đo keyword của reference
xuất hiện trong CÂU TRẢ LỜI (deterministic, không LLM). Đặt cạnh
`coverage_pct_strict` (đo trên chunk) để tách lỗi retrieval / generation /
judge — xem docstring của hàm.
"""

from slidegen_rag.evaluate.metrics import (
    calculate_answer_coverage,
    calculate_keyword_coverage_strict,
)


def test_answer_repeating_reference_scores_high():
    ref = "Trách nhiệm hình sự do Tòa án áp dụng đối với người phạm tội."
    assert calculate_answer_coverage(ref, ref) == 100.0


def test_unrelated_answer_scores_low():
    ref = "Trách nhiệm hình sự do Tòa án áp dụng đối với người phạm tội."
    cov = calculate_answer_coverage(ref, "Hợp đồng lao động xác định thời hạn.")
    assert cov < 30.0


def test_empty_answer_is_zero():
    assert calculate_answer_coverage("Nội dung tham chiếu nào đó.", "") == 0.0


def test_empty_reference_returns_full():
    # Giữ đúng quy ước của calculate_keyword_coverage_strict: không có
    # reference thì không có gì để bỏ sót.
    assert calculate_answer_coverage("", "câu trả lời bất kỳ") == 100.0


def test_matches_strict_coverage_when_answer_passed_as_single_chunk():
    # answer_coverage chỉ là strict coverage áp lên [answer] — giữ bất biến
    # này để hai con số luôn so sánh được trực tiếp với nhau.
    ref = "Năng lực hành vi dân sự đầy đủ được xác lập từ đủ 18 tuổi."
    ans = "Người thành niên từ đủ 18 tuổi có năng lực hành vi dân sự đầy đủ."
    assert calculate_answer_coverage(ref, ans) == calculate_keyword_coverage_strict(
        ref, [ans]
    )


def test_partial_answer_scores_between():
    ref = "Trách nhiệm hình sự do Tòa án áp dụng; trách nhiệm hành chính do cơ quan quản lý áp dụng."
    full = calculate_answer_coverage(ref, ref)
    half = calculate_answer_coverage(ref, "Trách nhiệm hình sự do Tòa án áp dụng.")
    assert 0.0 < half < full
