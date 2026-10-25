"""
Quản lý hội thoại nhiều lượt cho CRAG agent ở CLI (main.py).

Tách biệt 2 tầng state:
    - CRAG AgentState (agent/graph/state.py): ephemeral, reset hoàn toàn mỗi
      câu hỏi qua `make_initial_state()`.
    - ConversationState (`ConversationManager` ở đây): persistent trong tiến
      trình, tuỳ chọn persist qua Postgres (`repo`) — chat_history + node đã
      trích dẫn lượt gần nhất + tóm tắt lịch sử cũ.

Quyết định kiến trúc: KHÔNG dùng LangGraph checkpointer (MemorySaver/
SqliteSaver) — checkpointer giải quyết bài toán resume state của LangGraph
qua nhiều tiến trình, nhưng ConversationState ở đây KHÔNG phải AgentState:
nó nằm ngoài graph, không cần checkpoint từng bước, chỉ cần persist đầu/cuối
mỗi lượt. `repo` (SessionRepository, xem memory/repository.py) làm đúng việc
đó mà không kéo theo toàn bộ máy móc checkpointer. API server sau này không
chia sẻ instance `ConversationManager` giữa các request — mỗi request tạo
một instance mới qua `ConversationManager.resume()`, nạp lại state từ
Postgres; chính vì thế state phải nằm ở DB chứ không chỉ trong RAM.
"""

import logging
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from slidegen_rag.agent.models.llm import LLMModel
from slidegen_rag.agent.tools.tools import describe_node
from slidegen_rag.agent.utils.prompts import (
    CONTEXTUALIZE_SYSTEM_PROMPT,
    CONTEXTUALIZE_USER_PROMPT,
    HISTORY_COMPACT_SYSTEM_PROMPT,
    HISTORY_COMPACT_USER_PROMPT,
)
from slidegen_rag.memory.dto import CitedNodeRef, DocumentRef
from slidegen_rag.memory.repository import SessionRepository

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
    nhánh CRAG). Không thread-safe, không dùng chung giữa nhiều phiên.

    `repo=None` (mặc định) giữ hành vi thuần in-memory hôm nay KHÔNG đổi một
    byte — đây vừa là default vừa là guard hồi quy trong test. Truyền `repo`
    (+ `session_id` sau `resume()`, hoặc để `arecord_turn` tự tạo) để mỗi
    lượt được persist xuống Postgres."""

    def __init__(
        self,
        max_turns: int = 5,
        *,
        repo: SessionRepository | None = None,
        session_id: UUID | None = None,
        doc_ref: DocumentRef | None = None,
    ):
        self.max_turns = max_turns
        self.chat_history: list[
            dict
        ] = []  # [{"role": "user"|"assistant", "content": str, "_seq": int|None}]
        self._cited_last_turn: list[CitedNodeRef] = []
        self.history_summary: str | None = None
        self.repo = repo
        self.session_id = session_id
        self.doc_ref = doc_ref
        # DB rớt giữa phiên -> True, ngừng thử ghi lại mỗi lượt (xem
        # arecord_turn/maybe_compact) nhưng KHÔNG giết phiên đang chạy.
        self._degraded = False

    @classmethod
    async def resume(
        cls,
        repo: SessionRepository,
        session_id: UUID,
        doc_ref: DocumentRef | None,
        *,
        max_turns: int = 5,
    ) -> "ConversationManager":
        """Nạp lại một phiên đã lưu từ `repo`. `chat_history` chỉ chứa các
        lượt sau watermark `compacted_through_seq` (xem plan4.md B1.2) — cửa
        sổ context sau resume đúng bằng cửa sổ tiến trình trước đang giữ."""
        state = await repo.load_session(session_id)
        if state is None:
            raise ValueError(f"Session không tồn tại: {session_id}")
        conv = cls(
            max_turns=state.max_turns, repo=repo, session_id=state.id, doc_ref=doc_ref
        )
        conv.chat_history = [
            {"role": e.role, "content": e.content, "_seq": e.seq} for e in state.entries
        ]
        conv.history_summary = state.history_summary
        conv._cited_last_turn = list(state.last_cited)
        return conv

    @property
    def cited_node_ids_last_turn(self) -> list[int]:
        """Node index (Node.index trong cây `.pkl` ĐANG nạp) đã trích dẫn ở
        lượt trước. Lọc theo `doc_ref` hiện tại — nếu phiên được resume trên
        một `.pkl` khác (sha1 khác), citation cũ trỏ SAI node nên bị bỏ qua
        thay vì trả về id trùng ngẫu nhiên (xem plan4.md B1.1)."""
        if self.doc_ref is None:
            return [ref.node_index for ref in self._cited_last_turn]
        return [
            ref.node_index
            for ref in self._cited_last_turn
            if ref.doc_id == self.doc_ref.doc_id
            and ref.source_sha1 == self.doc_ref.source_sha1
        ]

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
        self,
        query: str,
        answer: str,
        document_node_ids: list | None,
        *,
        cited_refs: list[CitedNodeRef] | None = None,
    ) -> None:
        """Gọi sau khi có câu trả lời — cập nhật chat_history và node đã
        trích dẫn (lấy từ `traj.final_state["document_node_ids"]`). Thuần
        in-memory, KHÔNG persist — dùng `arecord_turn` cho việc đó.

        `cited_refs`: bỏ qua việc tự dựng `CitedNodeRef` từ `self.doc_ref`
        (đúng cho CLI — 1 phiên gắn đúng 1 `.pkl`) và dùng thẳng danh sách đã
        dựng sẵn — cần cho UI nhiều tài liệu, nơi mỗi node trích dẫn (node id
        đã offset trong cây gộp, xem `ui/merge.py`) có thể thuộc tài liệu
        khác nhau, không thể suy ra `doc_id`/`source_sha1` từ 1 `doc_ref`
        duy nhất."""
        self.chat_history.append({"role": "user", "content": query, "_seq": None})
        self.chat_history.append({"role": "assistant", "content": answer, "_seq": None})
        if cited_refs is not None:
            self._cited_last_turn = list(cited_refs)
            return
        doc_id = self.doc_ref.doc_id if self.doc_ref else ""
        source_sha1 = self.doc_ref.source_sha1 if self.doc_ref else ""
        self._cited_last_turn = [
            CitedNodeRef(doc_id=doc_id, source_sha1=source_sha1, node_index=nid)
            for nid in (document_node_ids or [])
            if nid is not None
        ]

    async def arecord_turn(
        self,
        query: str,
        standalone_query: str,
        answer: str,
        document_node_ids: list | None,
        *,
        cited_refs: list[CitedNodeRef] | None = None,
        assistant_meta: dict | None = None,
    ) -> None:
        """`record_turn()` rồi persist lượt vừa ghi qua `repo` (nếu có). Tiền
        tố `a` theo quy ước LangChain (`ainvoke`) — `record_turn` giữ nguyên
        sync vì thuần in-memory và 24 file test hiện có dựa vào đó.

        Lỗi ghi DB ở đây KHÔNG raise lại: đã trả lời xong (tốn token) mà giết
        phiên vì DB rớt là phá hoại — đánh dấu `_degraded` rồi tiếp tục
        in-memory cho các lượt sau (xem plan4.md B4)."""
        self.record_turn(query, answer, document_node_ids, cited_refs=cited_refs)
        if self.repo is None or self.session_id is None or self._degraded:
            return
        try:
            user_seq, assistant_seq = await self.repo.append_turn(
                self.session_id,
                raw_query=query,
                standalone_query=standalone_query,
                answer=answer,
                cited=self._cited_last_turn,
                assistant_meta=assistant_meta,
            )
        except Exception as e:
            logger.warning(
                f"[ConversationManager] Lưu lượt hội thoại thất bại, phiên chuyển "
                f"sang chế độ in-memory (không thử lưu lại): {e}"
            )
            self._degraded = True
            return
        self.chat_history[-2]["_seq"] = user_seq
        self.chat_history[-1]["_seq"] = assistant_seq

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
            return

        if self.repo is None or self.session_id is None or self._degraded:
            return
        last_seq = old_turns[-1].get("_seq")
        if last_seq is None:
            return  # lượt cuối bị nén chưa từng persist (vd. record_turn sync) — không có watermark để ghi
        try:
            await self.repo.save_compaction(
                self.session_id,
                summary=self.history_summary,
                compacted_through_seq=last_seq,
            )
        except Exception as e:
            logger.warning(
                f"[ConversationManager] save_compaction thất bại, phiên chuyển sang "
                f"chế độ in-memory (không thử lưu lại): {e}"
            )
            self._degraded = True
