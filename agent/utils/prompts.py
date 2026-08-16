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
# Generate — tổng hợp câu trả lời từ context
# ---------------------------------------------------------------------------

GENERATE_SYSTEM_PROMPT = """\
You are an expert assistant that answers questions based strictly on provided context.

Rules:
- Use ONLY information from the context below. Do not add outside knowledge.
- If the context does not contain enough information to answer, clearly state that.
- Be concise, accurate, and well-structured.
- Respond in the same language as the query.
"""

GENERATE_USER_PROMPT = """\
Context:
\"\"\"
{context}
\"\"\"

Query: {query}

Answer:\
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
