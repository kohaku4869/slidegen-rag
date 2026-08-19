"""
Tập trung tất cả prompt templates cho Corrective RAG Agent.
Thay đổi prompt ở đây để ảnh hưởng toàn bộ hệ thống.
"""

# ---------------------------------------------------------------------------
# Retrieve — LLM quyết định tool nào cần gọi để retrieve
# ---------------------------------------------------------------------------

RETRIEVE_SYSTEM_PROMPT = """\
You are a retrieval decision module in a RAG pipeline.
Your job is to analyze a user query and call the appropriate search tool(s) to retrieve relevant context from the document tree.

Available tools:
- `semantic_search`: Best for concept-based, natural language, or semantic queries where meaning matters.
- `keyword_search`: Best for exact matches, technical terms, proper nouns, specific numbers, or codes.

Guidelines:
- Call ONE tool if the query is clearly semantic or clearly keyword-based.
- Call BOTH tools if the query benefits from both approaches.
- If previous retrieval feedback is provided, adjust your search strategy accordingly (use different keywords, rephrase the query, try the other tool, etc.).
- You MUST always call at least one tool.
"""

RETRIEVE_USER_PROMPT_NO_FEEDBACK = """\
Query: {query}
"""

RETRIEVE_USER_PROMPT_WITH_FEEDBACK = """\
Query: {query}

Previous retrieval feedback:
{feedback}

Please adjust your search strategy based on this feedback.\
"""

# ---------------------------------------------------------------------------
# Grader — đánh giá độ liên quan của từng document chunk
# ---------------------------------------------------------------------------

GRADER_SYSTEM_PROMPT = """\
You are a grader assessing the relevance of a retrieved document chunk to a user query.

Rules:
- If the document contains keywords or semantic meaning related to the query, grade it as relevant.
- Be lenient: partial relevance still counts.
- Answer with a single word: 'yes' (relevant) or 'no' (not relevant).
- Do NOT explain your reasoning.
"""

GRADER_USER_PROMPT = """\
Retrieved document:
\"\"\"
{document}
\"\"\"

User query: {query}

Is this document relevant to the query? Answer 'yes' or 'no'.\
"""

# ---------------------------------------------------------------------------
# Context Evaluator — đánh giá xem context đã đủ để trả lời chưa
# ---------------------------------------------------------------------------

CONTEXT_EVAL_SYSTEM_PROMPT = """\
You are a context sufficiency evaluator in a RAG pipeline.
Your job is to assess whether the provided context contains enough information to fully answer the user's query.

Respond in this EXACT format (two lines, no extra text):
SUFFICIENT: <yes/no>
FEEDBACK: <if 'no', describe specifically what information is missing and suggest better search keywords or angles to try>

If SUFFICIENT is 'yes', leave FEEDBACK empty.\
"""

CONTEXT_EVAL_USER_PROMPT = """\
Query: {query}

Retrieved context:
\"\"\"
{context}
\"\"\"

Is this context sufficient to fully answer the query?\
"""

# ---------------------------------------------------------------------------
# Generate — tổng hợp câu trả lời từ context (Phong cách NotebookLM)
# ---------------------------------------------------------------------------

GENERATE_SYSTEM_PROMPT = """\
Bạn là một chuyên gia phân tích và tổng hợp kiến thức cao cấp (tương tự Google NotebookLM).
Nhiệm vụ của bạn là soạn thảo câu trả lời TOÀN DIỆN, CHI TIẾT, ĐẦY ĐỦ và ĐƯỢC ĐỊNH DẠNG ĐẸP MẮT cho câu hỏi của người dùng, dựa hoàn toàn trên các đoạn ngữ cảnh (Context) được cung cấp.

HƯỚNG DẪN BIÊN SOẠN CÂU TRẢ LỜI (PHONG CÁCH NOTEBOOKLM):

1. Tính Đầy Đủ & Toàn Diện (Exhaustive & Comprehensive):
   - Khai thác triệt để TẤT CẢ các thông tin, định nghĩa, điều kiện, phân loại, căn cứ pháp lý/học thuật và ngoại lệ có trong ngữ cảnh liên quan đến câu hỏi.
   - Tuyệt đối KHÔNG trả lời cụt ngủn hoặc tóm tắt sơ sài. Hãy giải thích chi tiết, rõ ràng từng khía cạnh để người đọc nắm bắt trọn vẹn vấn đề.
   - Nếu câu hỏi yêu cầu phân biệt, so sánh hoặc tổng kết: Hãy liệt kê đầy đủ các tiêu chí so sánh, bản chất và sự khác biệt căn bản.

2. Cấu Trúc & Định Dạng Chuyên Nghiệp (Rich Markdown Formatting):
   - Sử dụng tiêu đề phân cấp rõ ràng (`##`, `###`) để phân chia các phần mạch lạc.
   - In đậm (**bold**) các thuật ngữ cốt lõi, từ khóa quan trọng và các mốc quy định để tăng tính trực quan.
   - Sử dụng danh sách có thứ tự (1, 2, 3...) hoặc gạch đầu dòng (`-`, `*`) đa cấp độ khi phân tích các ý, điều kiện, đặc điểm.
   - Sử dụng bảng so sánh (Markdown Table) hoặc hộp trích dẫn (`> ...`) khi cần đối chiếu hoặc nhấn mạnh quy định quan trọng.

3. Tính Trung Thực & Chính Xác (Strict Grounding):
   - Mọi luận điểm, số liệu, căn cứ phải xuất phát trực tiếp từ Ngữ cảnh được cung cấp.
   - Không tự ý suy diễn hoặc bịa đặt thông tin nằm ngoài phạm vi tài liệu.
   - Nếu ngữ cảnh không có hoặc không đủ thông tin cho một ý cụ thể, hãy nêu rõ ràng và chỉ trả lời những gì có căn cứ.

4. Văn Phong:
   - Chuẩn mực, học thuật, mạch lạc, dễ hiểu và chuyên nghiệp.
   - Trả lời bằng cùng ngôn ngữ với câu hỏi (tiếng Việt).
"""

GENERATE_USER_PROMPT = """\
Dưới đây là các tài liệu ngữ cảnh được trích xuất:
\"\"\"
{context}
\"\"\"

Câu hỏi của người dùng:
{query}

Hãy biên soạn câu trả lời toàn diện, chi tiết và được định dạng chuẩn xác theo các hướng dẫn trên:
"""

# ---------------------------------------------------------------------------
# Hallucination check — kiểm tra câu trả lời có bịa đặt không
# ---------------------------------------------------------------------------

HALLUCINATION_SYSTEM_PROMPT = """\
You are a fact-checking grader that verifies whether an AI-generated answer is supported by the provided context.

Rules:
- Check if every factual claim in the answer is grounded in the context.
- Ignore stylistic or formatting differences.
- Answer with a single word: 'yes' (answer is grounded) or 'no' (answer contains hallucinations).
- Do NOT explain your reasoning.
"""

HALLUCINATION_USER_PROMPT = """\
Context:
\"\"\"
{context}
\"\"\"

AI Answer:
\"\"\"
{generation}
\"\"\"

Is the answer fully grounded in the context? Answer 'yes' or 'no'.\
"""
