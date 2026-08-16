from typing import TypedDict


class AgentState(TypedDict):
    """
    State cho Corrective RAG Agent (redesigned).

    Fields:
        query:              Câu hỏi gốc của user.
        tool_calls:         Danh sách tool calls được retrieve_node tạo ra,
                            tool_node sẽ đọc và thực thi.
        documents:          Các chunk tài liệu đã được graded (liên quan),
                            accumulate qua các lần retry.
        feedback:           Nhận xét từ context_evaluator_node về những gì
                            còn thiếu, để retrieve_node điều chỉnh lần sau.
        context_sufficient: True nếu context đã đủ để generate.
        generation:         Câu trả lời tổng hợp từ generate_node.
        retrieve_count:     Số lần đã chạy qua retrieve → tool → grade loop
                            (dùng để giới hạn vòng lặp, tránh infinite loop).
    """

    query: str
    tool_calls: list[dict]
    documents: list[str]
    feedback: str
    context_sufficient: bool
    generation: str
    retrieve_count: int