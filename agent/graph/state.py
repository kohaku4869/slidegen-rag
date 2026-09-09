from typing import TypedDict


class AgentState(TypedDict):
    """
    State cho Corrective RAG Agent (redesigned).

    Fields:
        query:              Câu hỏi gốc của user.
        tool_calls:         Danh sách tool calls được retrieve_node tạo ra,
                            tool_node sẽ đọc và thực thi.
        documents:          Các chunk tài liệu đã được graded (liên quan),
                            accumulate qua các lần retry. Chỉ grade_documents_node
                            được ghi field này (single-writer).
        pending_documents:  Chunk MỚI vừa lấy về ở vòng hiện tại, CHƯA được
                            grade. tool_node ghi đè field này mỗi vòng (không
                            tích luỹ — đã tích luỹ qua seen_fingerprints).
                            grade_documents_node đọc, chấm điểm rồi merge phần
                            đậu vào `documents`. Single-writer: tool_node.
        seen_fingerprints:  Hash các chunk đã từng thấy (mọi vòng, kể cả chunk
                            bị grade loại) để tool_node không đưa lại chunk cũ
                            vào pending_documents. Single-writer: tool_node.
        search_history:     Nhật ký các lần gọi tool: [{round, tool, query,
                            new_chunks, soft_failed}]. Dùng để retrieve_node
                            không lặp lại (tool, query) đã thử, và để
                            context_evaluator_node nêu rõ đã thử gì trong
                            feedback. Single-writer: tool_node.
        missing_aspects:    Danh sách khía cạnh câu hỏi mà context_evaluator_node
                            xác định là còn thiếu, để retrieve_node ưu tiên tìm
                            tiếp. Single-writer: context_evaluator_node.
        node_digest:        Danh mục gọn (id, layer, snippet, has_children) của
                            các node vừa retrieve, để retrieve_node biết node
                            nào có thể `expand_node` để lấy chi tiết.
                            Single-writer: tool_node.
        feedback:           Nhận xét từ context_evaluator_node về những gì
                            còn thiếu, để retrieve_node điều chỉnh lần sau.
        context_sufficient: True nếu context đã đủ để generate.
        generation:         Câu trả lời tổng hợp từ generate_node.
        retrieve_count:     Số lần đã chạy qua retrieve → tool → grade loop
                            (dùng để giới hạn vòng lặp, tránh infinite loop).

    Lưu ý thiết kế: graph này chạy TUẦN TỰ, không có fan-out (xem
    agent/graph/builder.py) — mỗi field trên có ĐÚNG MỘT node ghi, nên không
    cần Annotated reducer. Nếu sau này có ai thêm một writer thứ hai cho một
    field, phải thêm reducer tương ứng — nếu không update sẽ bị mất âm thầm.
    """

    query: str
    tool_calls: list[dict]
    documents: list[str]
    pending_documents: list[str]
    seen_fingerprints: list[str]
    search_history: list[dict]
    missing_aspects: list[str]
    node_digest: list[dict]
    feedback: str
    context_sufficient: bool
    generation: str
    retrieve_count: int
