"""
Ghi lại quỹ đạo (trajectory) thực thi của CRAG Agent: node nào chạy, gọi tool
gì với args gì, lặp mấy vòng retrieve (theo TỪNG facet), feedback của
context_evaluator từng vòng, latency & token/chi phí — tất cả những gì bị mất
khi chỉ dùng graph.ainvoke() (state cuối, không có lịch sử trung gian).

Cách hoạt động:
    graph.astream(state, stream_mode=["updates", "values"], subgraphs=True)
    được LangGraph dùng CHÍNH XÁC để cài đặt ainvoke() (giữ lại payload
    "values" cuối cùng có namespace=() làm kết quả trả về) — nên final_state
    ở đây LUÔN khớp graph.ainvoke(). "updates" cho biết thứ tự các node đã
    chạy (thấy rõ vòng lặp retrieve của từng facet).

    `subgraphs=True` là BẮT BUỘC kể từ khi graph có facet_subgraph (agent/
    graph/facet_subgraph.py) — mỗi bản facet_subgraph là 1 node cấp cha
    nhưng bên trong lại là 1 graph compiled riêng (retrieve→tool→
    grade_documents→context_evaluator, chạy N bản SONG SONG qua Send). Không
    có subgraphs=True, các bước bên trong hoàn toàn vô hình — mỗi mục
    "updates" chỉ thấy đúng 1 lần "facet_subgraph" hoàn tất, mất hết chi tiết
    tool call/round/feedback per-facet. Có subgraphs=True, mỗi item yielded
    là 3-tuple (namespace, mode, payload): namespace=() cho các bước cấp cha
    (decompose_facets, facet_subgraph tổng, collect_facets, generate);
    namespace=("facet_subgraph:<task_id>",) cho các bước BÊN TRONG 1 nhánh
    facet cụ thể. `<task_id>` không mang ý nghĩa gì (UUID nội bộ LangGraph)
    nên phải map namespace -> facet_label bằng THỨ TỰ xuất hiện lần đầu của
    mỗi namespace, khớp với thứ tự facet_queries mà fan_out_facets phát Send
    ra (LangGraph giữ đúng thứ tự list khi lập lịch task).

    TrajectoryCallbackHandler thu thập tool/LLM call trong lúc chạy — kể cả
    bên trong facet_subgraph, MIỄN LÀ facet_branch_node forward đúng `config`
    (mang theo callbacks) vào `.ainvoke()` của subgraph compiled (xem agent/
    graph/facet_subgraph.py). N facet chạy song song có thể phát callback từ
    nhiều task/thread khác nhau CÙNG LÚC — "drain" buffer của handler tại mỗi
    ranh giới node (bất kể namespace nào) vẫn đúng vì mỗi callback được gắn
    đúng ranh giới node của chính nó qua on_tool_start/on_llm_end (không dựa
    vào thứ tự toàn cục), TrajectoryCallbackHandler đã có threading.Lock sẵn
    cho việc này.

Sử dụng:
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.observability.trajectory import TrajectoryRecorder

    recorder = TrajectoryRecorder(graph, tags=["evaluate", topic])
    final_state, traj = recorder.run(make_initial_state(query), run_name="pldc:Q3")
    print(traj.render_text())
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from slidegen_rag.agent.observability.tracing import build_run_config, resolve_run_url
from slidegen_rag.agent.tools.tools import SOFT_FAILURE_PREFIXES
from slidegen_rag.config import AgentConfig, LangSmithConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    duration_s: float
    output_chars: int
    output_preview: str
    soft_failed: bool = False
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    model: str
    provider: str
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls_emitted: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class NodeStep:
    index: int
    name: str
    duration_s: float
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    update_summary: dict = field(default_factory=dict)
    error: Optional[str] = None
    facet: Optional[str] = None
    """Nhãn facet (facet_label) nếu step này chạy BÊN TRONG 1 nhánh
    facet_subgraph (namespace != ()) — None cho các bước cấp cha
    (decompose_facets, facet_subgraph tổng, collect_facets, generate)."""


@dataclass
class Trajectory:
    query: str
    started_at: datetime
    duration_s: float = 0.0
    steps: list[NodeStep] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    retrieve_count: int = 0
    max_retrieve: int = AgentConfig.MAX_RETRIEVE
    hit_max_retrieve: bool = False
    context_sufficient: bool = False
    facet_retrieve_counts: dict[str, int] = field(default_factory=dict)
    """retrieve_count CUỐI CÙNG của TỪNG facet (key = facet_label) — retrieve_count/
    hit_max_retrieve ở trên là MAX qua các facet này (giữ đúng ý nghĩa cũ "có
    facet nào chạm ngân sách chưa" từ thời graph còn 1 loop duy nhất)."""
    feedback_history: list[tuple[int, str]] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    run_id: Optional[str] = None
    langsmith_url: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def tool_usage_counts(self) -> Counter:
        counts: Counter = Counter()
        for step in self.steps:
            for tc in step.tool_calls:
                counts[tc.name] += 1
        return counts

    @property
    def loop_count(self) -> int:
        """Tổng số lượt "retrieve" TRÊN MỌI FACET cộng lại (không phải max
        theo facet — xem retrieve_count/facet_retrieve_counts cho con số
        đó)."""
        return sum(1 for step in self.steps if step.name == "retrieve")

    def render_text(self, *, title: Optional[str] = None) -> str:
        return render_trajectory(self, title=title)


# =============================================================================
# Callback handler — thu thập tool/LLM call, thread-safe
# =============================================================================


class TrajectoryCallbackHandler(BaseCallbackHandler):
    """
    Thu thập tool call & LLM call phát sinh trong lúc graph chạy.

    Không suy luận quan hệ cha-con qua parent_run_id (dễ vỡ khi đổi phiên bản
    LangGraph) — TrajectoryRecorder tự "drain" buffer này tại mỗi ranh giới
    node (qua stream_mode="updates") để gán đúng record vào đúng NodeStep.

    run_inline=True bắt buộc: nếu không, AsyncCallbackManager có thể đẩy
    handler sang executor khác, làm callback tới muộn hơn ranh giới node và
    lẫn sang bước kế tiếp (agent/nodes/tool_node.py chạy tool qua
    asyncio.to_thread nên các callback tool cũng có thể fire từ nhiều worker
    thread khác nhau — cần threading.Lock).
    """

    run_inline = True

    def __init__(self, *, tool_output_preview: int = 200):
        super().__init__()
        self._preview_len = tool_output_preview
        self._lock = threading.Lock()
        self._tool_starts: dict[UUID, tuple[str, dict, float]] = {}
        self._llm_starts: dict[UUID, tuple[str, str, float]] = {}
        self.tool_records: list[ToolCallRecord] = []
        self.llm_records: list[LLMCallRecord] = []

    def drain(self) -> tuple[list[ToolCallRecord], list[LLMCallRecord]]:
        """Lấy và xoá toàn bộ record đã tích luỹ (gọi tại mỗi ranh giới node)."""
        with self._lock:
            tools, self.tool_records = self.tool_records, []
            llms, self.llm_records = self.llm_records, []
        return tools, llms

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name", "?")
        with self._lock:
            self._tool_starts[run_id] = (name, dict(inputs or {}), time.perf_counter())

    def on_tool_end(self, output, *, run_id, **kwargs):
        now = time.perf_counter()
        with self._lock:
            name, args, t0 = self._tool_starts.pop(run_id, ("?", {}, now))
            text = str(output)
            soft_failed = any(text.startswith(p) for p in SOFT_FAILURE_PREFIXES)
            self.tool_records.append(
                ToolCallRecord(
                    name=name,
                    args=args,
                    duration_s=now - t0,
                    output_chars=len(text),
                    output_preview=text[: self._preview_len],
                    soft_failed=soft_failed,
                )
            )

    def on_tool_error(self, error, *, run_id, **kwargs):
        now = time.perf_counter()
        with self._lock:
            name, args, t0 = self._tool_starts.pop(run_id, ("?", {}, now))
            self.tool_records.append(
                ToolCallRecord(
                    name=name,
                    args=args,
                    duration_s=now - t0,
                    output_chars=0,
                    output_preview="",
                    soft_failed=False,
                    error=str(error),
                )
            )

    # ------------------------------------------------------------------
    # LLM / chat model callbacks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self, serialized, messages, *, run_id, metadata=None, **kwargs
    ):
        meta = metadata or {}
        model = meta.get("ls_model_name") or (serialized or {}).get("name") or "?"
        provider = meta.get("ls_provider") or "?"
        with self._lock:
            self._llm_starts[run_id] = (model, provider, time.perf_counter())

    def on_llm_end(self, response, *, run_id, **kwargs):
        now = time.perf_counter()
        with self._lock:
            model, provider, t0 = self._llm_starts.pop(run_id, ("?", "?", now))

        usage: dict = {}
        tool_calls_emitted: list[str] = []
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg is not None:
                usage = getattr(msg, "usage_metadata", None) or {}
                tool_calls_emitted = [
                    tc.get("name") for tc in (getattr(msg, "tool_calls", None) or [])
                ]
        except (IndexError, AttributeError, KeyError):
            pass

        with self._lock:
            self.llm_records.append(
                LLMCallRecord(
                    model=model,
                    provider=provider,
                    duration_s=now - t0,
                    input_tokens=usage.get("input_tokens", 0) or 0,
                    output_tokens=usage.get("output_tokens", 0) or 0,
                    total_tokens=usage.get("total_tokens", 0) or 0,
                    tool_calls_emitted=tool_calls_emitted,
                )
            )

    def on_llm_error(self, error, *, run_id, **kwargs):
        now = time.perf_counter()
        with self._lock:
            model, provider, t0 = self._llm_starts.pop(run_id, ("?", "?", now))
            self.llm_records.append(
                LLMCallRecord(
                    model=model,
                    provider=provider,
                    duration_s=now - t0,
                    error=str(error),
                )
            )


# =============================================================================
# Recorder
# =============================================================================


def _summarize_update(update: dict) -> dict:
    """Tóm tắt state update của 1 node: list -> độ dài (int, để tính delta ở
    renderer), string dài -> cắt ngắn. Không giữ payload thô (documents đầy
    đủ đã có sẵn trong chunks.txt, không cần nhân đôi trong trajectory)."""
    summary: dict = {}
    for k, v in (update or {}).items():
        if isinstance(v, list):
            summary[k] = len(v)
        elif isinstance(v, str) and len(v) > 240:
            summary[k] = v[:240] + "…"
        else:
            summary[k] = v
    return summary


class TrajectoryRecorder:
    """Chạy graph MỘT LẦN qua astream(stream_mode=["updates","values"]) và
    ghi lại toàn bộ trajectory — thay thế trực tiếp cho graph.ainvoke()."""

    def __init__(
        self,
        graph,
        *,
        run_name_prefix: str = "crag",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        on_step: Optional[Callable[[NodeStep], None]] = None,
    ):
        self.graph = graph
        self.run_name_prefix = run_name_prefix
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})
        self.on_step = on_step

    async def arun(
        self,
        initial_state: dict,
        *,
        run_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[dict, Trajectory]:
        from slidegen_rag.agent.models.llm import LLMModel

        LLMModel.reset_usage()

        handler = TrajectoryCallbackHandler(
            tool_output_preview=LangSmithConfig.TRAJECTORY_TOOL_OUTPUT_PREVIEW
        )
        merged_tags = self.tags + list(tags or [])
        merged_metadata = {**self.metadata, **(metadata or {})}
        query_preview = str(initial_state.get("query", ""))[:60]
        final_run_name = run_name or f"{self.run_name_prefix}:{query_preview}"

        config, tracer = build_run_config(
            final_run_name,
            tags=merged_tags,
            metadata=merged_metadata,
            callbacks=[handler],
        )

        started_at = datetime.now()
        t_start = time.perf_counter()
        t_prev = t_start
        steps: list[NodeStep] = []
        final_state: dict = dict(initial_state)
        error: Optional[str] = None

        # Map namespace ("facet_subgraph:<task_id>",) -> facet_label, theo
        # THỨ TỰ xuất hiện lần đầu của mỗi namespace (khớp thứ tự Send mà
        # fan_out_facets phát ra — xem module docstring). facet_labels được
        # điền ngay khi "updates" của node "decompose_facets" (namespace=())
        # tới, TRƯỚC khi bất kỳ namespace facet nào xuất hiện.
        facet_labels: list[str] = []
        namespace_to_label: dict[str, str] = {}

        try:
            async for namespace, mode, payload in self.graph.astream(
                initial_state,
                stream_mode=["updates", "values"],
                subgraphs=True,
                config=config,
            ):
                now = time.perf_counter()
                if mode == "updates":
                    for node_name, node_output in (payload or {}).items():
                        if node_name == "decompose_facets" and not namespace:
                            fq = (node_output or {}).get("facet_queries") or []
                            facet_labels = [f.get("label", "?") for f in fq]

                        facet_label: Optional[str] = None
                        if namespace:
                            ns_key = namespace[0]
                            if ns_key not in namespace_to_label:
                                idx = len(namespace_to_label)
                                namespace_to_label[ns_key] = (
                                    facet_labels[idx]
                                    if idx < len(facet_labels)
                                    else f"facet_{idx + 1}"
                                )
                            facet_label = namespace_to_label[ns_key]

                        tool_recs, llm_recs = handler.drain()
                        step = NodeStep(
                            index=len(steps) + 1,
                            name=node_name,
                            duration_s=now - t_prev,
                            tool_calls=tool_recs,
                            llm_calls=llm_recs,
                            update_summary=_summarize_update(node_output),
                            facet=facet_label,
                        )
                        steps.append(step)
                        t_prev = now
                        if self.on_step is not None:
                            try:
                                self.on_step(step)
                            except Exception:
                                logger.debug(
                                    "[trajectory] on_step callback lỗi, bỏ qua",
                                    exc_info=True,
                                )
                elif mode == "values":
                    # Chỉ cập nhật final_state ở namespace TOP-LEVEL (()) —
                    # "values" với namespace khác là final FacetState riêng
                    # của 1 nhánh facet, không phải state cha (AgentState).
                    if not namespace:
                        final_state = payload
        except Exception as e:
            logger.error(f"[trajectory] Agent chạy lỗi: {e}")
            error = str(e)

        duration_s = time.perf_counter() - t_start

        # retrieve_count CUỐI CÙNG của TỪNG facet (increment đơn điệu trong
        # chính facet đó, nên giá trị CUỐI = số vòng loop thật của facet đó).
        facet_retrieve_counts: dict[str, int] = {}
        for s in steps:
            if s.name == "retrieve" and isinstance(
                s.update_summary.get("retrieve_count"), int
            ):
                key = s.facet or "?"
                facet_retrieve_counts[key] = s.update_summary["retrieve_count"]

        final_retrieve_count = (
            max(facet_retrieve_counts.values())
            if facet_retrieve_counts
            else final_state.get("retrieve_count", 0)
        )

        # context_sufficient toàn cục = MỌI facet đều sufficient (facet_sufficiency
        # do collect_facets_node build trên final_state cấp cha — thay thế
        # hẳn field context_sufficient toàn cục cũ, không còn tồn tại trên
        # AgentState mới).
        facet_sufficiency = final_state.get("facet_sufficiency") or []
        context_sufficient = bool(facet_sufficiency) and all(
            f.get("sufficient", False) for f in facet_sufficiency
        )

        # Lịch sử feedback theo từng vòng retrieve, TÁCH RIÊNG theo facet (N
        # facet chạy song song nên "vòng lặp hiện tại" phải theo dõi độc lập
        # cho mỗi facet, không dùng chung 1 biến đếm toàn cục).
        feedback_history: list[tuple[int, str]] = []
        loop_no_by_facet: dict[Optional[str], int] = {}
        for s in steps:
            if s.name == "retrieve":
                rc = s.update_summary.get("retrieve_count")
                loop_no_by_facet[s.facet] = (
                    rc if isinstance(rc, int) else loop_no_by_facet.get(s.facet, 0) + 1
                )
            if s.name == "context_evaluator":
                fb = s.update_summary.get("feedback")
                if fb:
                    feedback_history.append((loop_no_by_facet.get(s.facet, 0), fb))

        token_usage = LLMModel.get_usage_summary()

        langsmith_url = None
        if tracer is not None and LangSmithConfig.RESOLVE_RUN_URL:
            langsmith_url = resolve_run_url(tracer)
        run_id = None
        if tracer is not None:
            latest_run = getattr(tracer, "latest_run", None)
            rid = getattr(latest_run, "id", None)
            run_id = str(rid) if rid else None

        traj = Trajectory(
            query=str(initial_state.get("query", "")),
            started_at=started_at,
            duration_s=duration_s,
            steps=steps,
            final_state=final_state,
            retrieve_count=final_retrieve_count,
            max_retrieve=AgentConfig.MAX_RETRIEVE,
            hit_max_retrieve=(
                final_retrieve_count >= AgentConfig.MAX_RETRIEVE and not context_sufficient
            ),
            context_sufficient=context_sufficient,
            facet_retrieve_counts=facet_retrieve_counts,
            feedback_history=feedback_history,
            token_usage=token_usage,
            run_id=run_id,
            langsmith_url=langsmith_url,
            error=error,
            metadata=merged_metadata,
        )
        return final_state, traj

    def run(self, initial_state: dict, **kwargs: Any) -> tuple[dict, Trajectory]:
        """Bọc asyncio.run cho code đồng bộ (evaluate.py)."""
        return asyncio.run(self.arun(initial_state, **kwargs))


# =============================================================================
# Renderer — text tiếng Việt
# =============================================================================

_NODE_LABELS = {
    "retrieve": "retrieve",
    "tool": "tool",
    "grade_documents": "grade_documents",
    "context_evaluator": "context_evaluator",
    "generate": "generate",
}


def _fmt_args(args: dict) -> str:
    parts = [f"{k}={v!r}" for k, v in (args or {}).items()]
    return "{" + ", ".join(parts) + "}"


def _render_step(step: NodeStep, ctx: dict) -> list[str]:
    """Render 1 NodeStep. `ctx` là state có thể thay đổi giữa các bước:
    doc_count (số documents ở bước trước) và retrieve_loop (vòng retrieve
    hiện tại) — dùng để tính delta / hiển thị "vòng #n". `ctx` PHẢI là context
    RIÊNG của facet này (hoặc riêng của cấp cha) — xem render_trajectory():
    N facet chạy song song nên steps của chúng interleave trong traj.steps,
    dùng chung 1 ctx sẽ tính delta/vòng lặp sai (trộn facet A với facet B)."""
    lines = []
    label = _NODE_LABELS.get(step.name, step.name)
    header = f"[{step.index}] {label:<18} ({step.duration_s:.2f}s)"
    if step.facet:
        header += f"  [facet: {step.facet}]"

    if step.name == "retrieve":
        rc = step.update_summary.get("retrieve_count")
        if isinstance(rc, int):
            ctx["retrieve_loop"] = rc
            header += f"  vòng #{rc}"
            if rc > 1:
                header += "  (có feedback)"
    lines.append(header)

    for llm in step.llm_calls:
        if llm.error:
            lines.append(f"    · LLM  {llm.model} ({llm.provider})  LỖI: {llm.error}")
            continue
        tc_str = ""
        if llm.tool_calls_emitted:
            tc_str = f" → tool_calls: {', '.join(llm.tool_calls_emitted)}"
        lines.append(
            f"    · LLM  {llm.model} ({llm.provider})  {llm.duration_s:.2f}s "
            f"in={llm.input_tokens} out={llm.output_tokens}{tc_str}"
        )

    if step.name == "grade_documents" and len(step.llm_calls) > 1:
        # Gộp lại thành 1 dòng tổng hợp thay vì liệt kê từng lượt grader —
        # 3 vòng x ~50 call sẽ làm file vô dụng nếu in từng dòng.
        lines = lines[:1]  # bỏ các dòng LLM riêng lẻ vừa thêm ở trên
        n = len(step.llm_calls)
        total_dur = sum(c.duration_s for c in step.llm_calls)
        total_in = sum(c.input_tokens for c in step.llm_calls)
        total_out = sum(c.output_tokens for c in step.llm_calls)
        errors = sum(1 for c in step.llm_calls if c.error)
        err_str = f", {errors} lỗi" if errors else ""
        model = step.llm_calls[0].model
        lines.append(
            f"    · LLM  {model} × {n} lượt  tổng {total_dur:.2f}s "
            f"in={total_in} out={total_out}{err_str}"
        )

    for tc in step.tool_calls:
        if tc.error:
            status = f"LỖI: {tc.error}"
        elif tc.soft_failed:
            status = "MỀM-LỖI (tool trả lỗi dạng chuỗi, không raise)"
        else:
            status = "OK"
        lines.append(
            f"    · TOOL {tc.name:<16} args={_fmt_args(tc.args)}  "
            f"{tc.duration_s:.2f}s  out={tc.output_chars:,} ký tự  {status}"
        )

    # ── cập nhật state (delta hoá cho documents) ──
    upd = step.update_summary

    if step.name == "tool" and "pending_documents" in upd:
        n_pending = upd["pending_documents"]
        ctx["pending"] = n_pending
        lines.append(
            f"    → cập nhật: pending_documents=list[{n_pending}]  "
            f"(+{n_pending} chunk mới, đã lọc trùng xuyên vòng)"
        )

    if "documents" in upd:
        new_count = upd["documents"]
        prev_count = ctx.get("doc_count", 0)
        if step.name == "grade_documents":
            pending = ctx.get("pending", 0)
            kept = new_count - prev_count
            loai = max(pending - kept, 0)
            lines.append(
                f"    → cập nhật: documents=list[{new_count}]  "
                f"(chấm {pending} chunk mới: giữ {kept}, loại {loai})"
            )
        else:
            lines.append(f"    → cập nhật: documents=list[{new_count}]")
        ctx["doc_count"] = new_count

    if step.name == "context_evaluator":
        suff = upd.get("context_sufficient")
        fb = upd.get("feedback") or ""
        lines.append(f"    → context_sufficient={suff}")
        if fb:
            lines.append(f'    → feedback: "{fb}"')
        loop = ctx.get("retrieve_loop", 0)
        if suff:
            lines.append("    → định tuyến: generate")
        elif loop >= AgentConfig.MAX_RETRIEVE:
            lines.append(
                f"    → định tuyến: generate  (hết lượt retrieve, best-effort {loop}/{AgentConfig.MAX_RETRIEVE})"
            )
        else:
            lines.append(
                f"    → định tuyến: retrieve  (retrieve_count={loop} < MAX_RETRIEVE={AgentConfig.MAX_RETRIEVE})"
            )

    if step.name == "retrieve" and "tool_calls" in upd:
        lines.append(
            f"    → cập nhật: tool_calls=list[{upd['tool_calls']}], retrieve_count={upd.get('retrieve_count', '?')}"
        )

    if step.name == "generate":
        gen_val = upd.get("generation")
        if isinstance(gen_val, str):
            lines.append(
                f"    → cập nhật: generation sinh xong ({len(gen_val)} ký tự đã tóm tắt)"
            )

    if step.error:
        lines.append(f"    ⚠ LỖI NODE: {step.error}")

    return lines


def render_trajectory(traj: Trajectory, *, title: Optional[str] = None) -> str:
    """Kết xuất Trajectory ra text tiếng Việt, người-đọc-được."""
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("")

    lines.append(f"Câu hỏi : {traj.query}")
    lines.append(f"Bắt đầu : {traj.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    end_desc = (
        "ĐỦ NGỮ CẢNH"
        if traj.context_sufficient
        else (
            f"HẾT LƯỢT RETRIEVE (best-effort, context_sufficient=False, "
            f"{traj.retrieve_count}/{traj.max_retrieve})"
            if traj.hit_max_retrieve
            else "CHƯA ĐỦ NGỮ CẢNH"
        )
    )
    lines.append(
        f"Tổng thời gian: {traj.duration_s:.1f}s | Số bước node: {len(traj.steps)} "
        f"| Vòng lặp retrieve: {traj.retrieve_count}/{traj.max_retrieve}"
    )
    if traj.facet_retrieve_counts:
        facet_loops_str = ", ".join(
            f"{label}={n}" for label, n in traj.facet_retrieve_counts.items()
        )
        lines.append(f"Vòng lặp theo facet: {facet_loops_str}")
    lines.append(f"Kết thúc: {end_desc}")

    usage = traj.token_usage or {}
    if usage:
        vnd = usage.get("total_cost_vnd", 0.0)
        usd = usage.get("total_cost_usd", 0.0)
        # Gemini-only — đây là phần TÍNH PHÍ thật, tách khỏi local (miễn phí)
        # để chi phí ước tính không bị pha loãng bởi token của local model
        # (xem TokenTracker ở agent/models/llm.py). Fallback về khoá tổng
        # gộp cũ (input_tokens/...) cho token_usage được ghi trước bản sửa
        # này (không có khoá gemini_*).
        gem_in = usage.get("gemini_input_tokens", usage.get("input_tokens", 0))
        gem_out = usage.get("gemini_output_tokens", usage.get("output_tokens", 0))
        gem_tot = usage.get("gemini_total_tokens", usage.get("total_tokens", 0))
        lines.append(
            f"Tokens Gemini: in={gem_in:,} out={gem_out:,} tổng={gem_tot:,} "
            f"| Chi phí ước tính: ${usd:.6f} USD (~{vnd:.2f} VNĐ)"
        )
        local_tot = usage.get("local_total_tokens", 0)
        if local_tot:
            lines.append(
                f"Tokens Local (miễn phí): tổng={local_tot:,} "
                f"({usage.get('local_calls', 0)} lượt)"
            )

    if traj.run_id:
        line = f"LangSmith: project={LangSmithConfig.PROJECT} | run_id={traj.run_id}"
        lines.append(line)
        if traj.langsmith_url:
            lines.append(f"           {traj.langsmith_url}")
    else:
        lines.append("LangSmith: (tắt)")

    if traj.error:
        lines.append(f"LỖI AGENT: {traj.error}")

    lines.append("")
    lines.append("── QUỸ ĐẠO THỰC THI " + "─" * 55)

    # 1 ctx RIÊNG cho mỗi facet (key=facet_label, None=cấp cha) — N facet chạy
    # song song nên steps của chúng interleave trong traj.steps; dùng chung 1
    # ctx sẽ tính sai delta documents / vòng lặp hiện tại (trộn lẫn facet A
    # với facet B khi steps của 2 facet xen kẽ nhau trong danh sách).
    ctx_by_facet: dict[Optional[str], dict] = {}
    for step in traj.steps:
        ctx = ctx_by_facet.setdefault(step.facet, {"doc_count": 0, "retrieve_loop": 0})
        lines.extend(_render_step(step, ctx))
        lines.append("")

    lines.append("── THỐNG KÊ " + "─" * 63)
    tool_counts = traj.tool_usage_counts
    total_tool_calls = sum(tool_counts.values())
    tool_errors = sum(
        1 for s in traj.steps for tc in s.tool_calls if tc.error or tc.soft_failed
    )
    if tool_counts:
        tool_str = ", ".join(f"{name} ×{n}" for name, n in tool_counts.items())
        lines.append(
            f"Tool đã gọi     : {tool_str}  (tổng {total_tool_calls} lượt, {tool_errors} lỗi)"
        )
    else:
        lines.append("Tool đã gọi     : (không có)")

    # provider thật do BaseChatModel._get_ls_params() gán qua metadata:
    # ChatGoogleGenerativeAI -> "google_genai", ChatOpenAI (local/vLLM) -> "openai"
    all_llm = [c for s in traj.steps for c in s.llm_calls]
    gemini_calls = sum(1 for c in all_llm if c.provider == "google_genai")
    local_calls = sum(1 for c in all_llm if c.provider == "openai")
    other_calls = len(all_llm) - gemini_calls - local_calls
    other_str = f", khác {other_calls}" if other_calls else ""
    lines.append(
        f"Số lượt gọi LLM : {len(all_llm)} (Gemini {gemini_calls}, Local {local_calls}{other_str})"
    )

    final_docs = traj.final_state.get("documents", [])
    lines.append(
        f"Chunk cuối cùng : {len(final_docs) if isinstance(final_docs, list) else 0}"
    )

    slowest = sorted(traj.steps, key=lambda s: s.duration_s, reverse=True)[:3]
    if slowest:
        slow_str = ", ".join(f"{s.name} ({s.duration_s:.2f}s)" for s in slowest)
        lines.append(f"Node chậm nhất  : {slow_str}")

    lines.append(f"Lỗi             : {traj.error or '(không có)'}")

    return "\n".join(lines)


# =============================================================================
# I/O — song song save_chunks/load_chunks trong evaluate/evaluate.py
# =============================================================================


def render_trajectories_file(trajs: dict[int, Trajectory], topic: str) -> str:
    """Ghép nhiều Trajectory (theo số câu hỏi) thành 1 file, block
    '=== CÂU HỎI N ===' giống hệt chunks.txt."""
    blocks = []
    for num in sorted(trajs.keys()):
        traj = trajs[num]
        blocks.append(f"=== CÂU HỎI {num} ===\n\n{render_trajectory(traj)}")
    return "\n\n".join(blocks) + "\n"


def save_trajectories(path: Path, trajs: dict[int, Trajectory], topic: str) -> None:
    """Ghi trajectory.txt (UTF-8), tạo thư mục cha nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_trajectories_file(trajs, topic), encoding="utf-8")
