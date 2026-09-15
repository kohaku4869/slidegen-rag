"""
Quản lý hội thoại nhiều lượt cho CRAG agent ở CLI (main.py).

Tách biệt 2 tầng state:
    - CRAG AgentState (agent/graph/state.py): ephemeral, reset hoàn toàn mỗi
      câu hỏi qua `make_initial_state()`.
    - ConversationState (`ConversationManager` ở đây): persistent, chỉ giữ
      chat_history + node đã trích dẫn lượt gần nhất + tóm tắt lịch sử cũ.

Quyết định kiến trúc: KHÔNG dùng LangGraph checkpointer (MemorySaver/
SqliteSaver) — `main.py` hiện chạy 1
process/phiên duy nhất và đã giữ state ngay trong Python, nên checkpointer
(giải quyết bài toán persist qua nhiều tiến trình/API) là thừa máy móc so
với nhu cầu thực tế. Nếu sau này có API server cần resume qua nhiều tiến
trình, đó là một việc bổ sung tách biệt, không phải sửa lại class này.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.tools.tools import describe_node
from slidegen_rag.agent.utils.prompts import (
    CONTEXTUALIZE_SYSTEM_PROMPT,
    CONTEXTUALIZE_USER_PROMPT,
    HISTORY_COMPACT_SYSTEM_PROMPT,
    HISTORY_COMPACT_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# temperature=0 cho cả 2 việc (viết lại câu hỏi, tóm tắt) — cần output ổn định,
# không cần sáng tạo.
_contextualize_llm = LLMModel(temperature=0.0, max_tokens=256)
_compact_llm = LLMModel(temperature=0.0, max_tokens=512)


def _format_turns(turns: list[dict]) -> str:
    if not turns:
        return "(không có)"
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {t.get('content', '')}")
    return "\n".join(lines)


class ConversationManager:
    """1 instance / phiên chat tương tác (`main.py::interactive_loop`, chỉ
    nhánh CRAG). Không thread-safe, không dùng chung giữa nhiều phiên."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.chat_history: list[
            dict
        ] = []  # [{"role": "user"|"assistant", "content": str}]
        self.cited_node_ids_last_turn: list[int] = []
        self.history_summary: Optional[str] = None

    async def contextualize(self, new_query: str) -> str:
        """
        Viết lại câu hỏi follow-up thành câu độc lập dựa trên vài lượt gần
        nhất. Bỏ qua (trả nguyên câu) nếu chưa có lịch sử —
        không tốn thêm 1 LLM call cho câu hỏi đầu tiên của phiên.
        """
        if not self.chat_history:
            return new_query

        history_text = _format_turns(self.chat_history[-3:])
        if self.history_summary:
            history_text = (
                f"(Tóm tắt các lượt trước đó: {self.history_summary})\n{history_text}"
            )

        messages = [
            SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
            HumanMessage(
                content=CONTEXTUALIZE_USER_PROMPT.format(
                    history=history_text, new_query=new_query
                )
            ),
        ]
        try:
            response = await _contextualize_llm.ainvoke(messages)
            rewritten = response.content.strip()
            if rewritten and rewritten != new_query:
                logger.info(
                    f"[ConversationManager] Contextualized: '{new_query[:60]}' -> '{rewritten[:60]}'"
                )
            return rewritten or new_query
        except Exception as e:
            logger.warning(
                f"[ConversationManager] contextualize thất bại: {e} — dùng câu hỏi gốc"
            )
            return new_query

    def seed_drill_candidates(self) -> list[dict]:
        """
        Dựng drill_candidates ban đầu từ các node đã trích dẫn ở lượt trước,
        để retrieve_node của câu hỏi MỚI ưu tiên đào sâu (Drill-down) thay vì
        Discovery lại từ đầu — dùng cùng schema/reason
        ("need_detail"/"need_siblings"/"need_context") mà grade_documents_node
        tạo ra trong 1 lượt, chỉ khác nguồn gốc (lượt trước thay vì vòng
        grading hiện tại).
        """
        candidates = []
        for node_id in self.cited_node_ids_last_turn:
            info = describe_node(node_id)
            if info is None:
                continue
            if info.get("has_children"):
                reason = "need_detail"
            elif info.get("n_siblings", 0) >= 2:
                reason = "need_siblings"
            elif info.get("has_parent"):
                reason = "need_context"
            else:
                continue  # node lá cô lập, không đủ anh em — không có gì để đào tiếp
            candidates.append({"node_id": node_id, "content": "", "reason": reason})
        return candidates

    def record_turn(
        self, query: str, answer: str, document_node_ids: Optional[list]
    ) -> None:
        """Gọi sau khi có câu trả lời — cập nhật chat_history và node đã
        trích dẫn (lấy từ `traj.final_state["document_node_ids"]`)."""
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})
        self.cited_node_ids_last_turn = [
            nid for nid in (document_node_ids or []) if nid is not None
        ]

    async def maybe_compact(self) -> None:
        """
        Cuộn các lượt cũ nhất thành 1 đoạn tóm tắt khi vượt `max_turns`
        — 1 LLM call duy nhất cho mọi lượt bị cuộn, không
        phải mỗi lượt 1 call.
        """
        max_entries = self.max_turns * 2  # mỗi lượt = 1 entry user + 1 entry assistant
        if len(self.chat_history) <= max_entries:
            return

        old_turns = self.chat_history[:-max_entries]
        recent_turns = self.chat_history[-max_entries:]

        messages = [
            SystemMessage(content=HISTORY_COMPACT_SYSTEM_PROMPT),
            HumanMessage(
                content=HISTORY_COMPACT_USER_PROMPT.format(
                    turns=_format_turns(old_turns),
                    previous_summary=self.history_summary or "(không có)",
                )
            ),
        ]
        try:
            response = await _compact_llm.ainvoke(messages)
            self.history_summary = response.content.strip()
            self.chat_history = recent_turns
            logger.info(
                f"[ConversationManager] Đã nén {len(old_turns)} entry lịch sử cũ thành summary"
            )
        except Exception as e:
            logger.warning(
                f"[ConversationManager] maybe_compact thất bại: {e} — giữ nguyên lịch sử"
            )
