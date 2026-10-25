"""
Điều phối một lượt đánh giá chủ đề: lấy câu trả lời (chạy agent, đọc cache,
chỉ chạy lại generate, hoặc chạy một phiên hội thoại nhiều lượt có short-term
memory) rồi chấm điểm và gom thành bảng chi tiết + summary.

Mỗi lần gọi `run_evaluation_for_topic` tạo MỘT run mới, tự chứa, trong
`data/test/<topic>/runs/<run_id>/` (xem plan5.md Phần C) — không còn ghi đè
answer.txt/chunks.txt/trajectory.txt phẳng như layout cũ.
"""

import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from slidegen_rag.evaluate import run_config as run_config_mod
from slidegen_rag.evaluate.io import (
    load_answers,
    load_chunks,
    load_generate_state,
    load_questions,
    load_references,
    load_trajectory_texts,
    save_answers,
    save_chunks,
    save_generate_state,
)
from slidegen_rag.evaluate.judge import (
    create_evaluator_llm,
    evaluate_single_answer_with_judge,
    judge_tracker,
)
from slidegen_rag.evaluate.metrics import (
    calculate_answer_coverage,
    calculate_keyword_coverage,
    calculate_keyword_coverage_strict,
)
from slidegen_rag.evaluate.runs import (
    RunPaths,
    resolve_source_run,
    write_latest_pointer,
)
from slidegen_rag.evaluate.runs import (
    question_file as runs_question_file,
)
from slidegen_rag.evaluate.runs import (
    reference_file as runs_reference_file,
)
from slidegen_rag.evaluate.settings import (
    DATA_INDEX_DIR,
    GENERATE_STATE_FILENAME,
)

logger = logging.getLogger(__name__)

# GEMINI_MODEL env override giống hệt agent/models/llm.py::LLMModel — đọc lại
# ở đây (thay vì import LLMConfig) để ghi đúng model THẬT SỰ được dùng vào
# run_config.json ngay cả khi biến môi trường override giá trị mặc định.
def _current_gemini_model() -> str:
    from slidegen_rag.config import LLMConfig

    return os.getenv("GEMINI_MODEL") or LLMConfig.MODEL


def _current_local_base_url() -> Optional[str]:
    return os.getenv("LLM_BASE_URL", "").strip() or None


def _add_token_summaries(a: dict, b: dict) -> dict:
    """Cộng 2 dict cùng shape TokenTracker summary (xem
    agent/models/llm.py::TokenTracker) — mọi field đều tuyến tính theo số
    token/lượt gọi nên cộng field-wise là ĐÚNG, không cần cộng lại từ counter
    thô."""
    keys = set(a) | set(b)
    return {k: a.get(k, 0) + b.get(k, 0) for k in keys}


_EMPTY_TOKEN_SUMMARY = {
    "calls": 0,
    "gemini_calls": 0,
    "local_calls": 0,
    "gemini_input_tokens": 0,
    "gemini_output_tokens": 0,
    "gemini_reasoning_tokens": 0,
    "gemini_total_tokens": 0,
    "local_input_tokens": 0,
    "local_output_tokens": 0,
    "local_total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "reasoning_tokens": 0,
    "total_cost_usd": 0.0,
    "total_cost_vnd": 0.0,
}


@dataclass
class AnswerBatch:
    """Kết quả của một lần lấy câu trả lời (chạy agent / đọc cache / phiên
    multi-turn) — gộp mọi thông tin `run_evaluation_for_topic` cần để ghi
    `run_config.json` và (nếu multi-turn) `conversation.json`/
    `turn_metrics.json`. Rộng hơn `tuple[list[dict], bool]` nêu trong
    plan5.md B1 (pseudocode ở đó lược bớt để dễ đọc) — chỉ dùng nội bộ trong
    package này nên đổi hình dạng không ảnh hưởng call-site bên ngoài."""

    results: list[dict]
    was_cached: bool
    source_run_id: Optional[str]
    generate_provider: str
    index_fingerprint: dict = field(
        default_factory=lambda: {"path": None, "source_sha1": None, "pkl_sha1": None}
    )
    conversation_log: Optional[dict] = None


def _resolve_index_path(topic: str):
    v2_path = DATA_INDEX_DIR / f"{topic}_v2.pkl"
    return v2_path if v2_path.exists() else DATA_INDEX_DIR / f"{topic}.pkl"


def get_answers_and_chunks_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
    lowcost: bool = False,
    multi_turn: bool = False,
    max_turns: int = 5,
    run_dir: Optional[RunPaths] = None,
) -> AnswerBatch:
    """
    Lấy danh sách câu trả lời và chunks cho chủ đề — 3 nhánh:

    1. `multi_turn=True`: LUÔN chạy phiên mới (không cache — mỗi phiên có
       lịch sử riêng, xem `_run_multi_turn_session`).
    2. Cache hit: run nguồn đủ câu VÀ `config_fingerprint` khớp cấu hình sắp
       chạy (xem evaluate/run_config.py) — copy artifact sang `run_dir` để
       run vẫn tự chứa (xem plan5.md C1).
    3. Chạy đầy đủ (retrieve→...→generate) từng câu qua CRAG agent.

    Args:
        lowcost: True → generate_node dùng model tự host (LLM_BASE_URL) thay
            Gemini. Các node khác vẫn Gemini.
        run_dir: `RunPaths` đã tạo sẵn (thường do `run_evaluation_for_topic`
            dựng ở đầu run) — tự tạo mới nếu không truyền vào.

    Returns:
        AnswerBatch
    """
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.nodes.generate_node import set_generate_provider
    from slidegen_rag.agent.observability.trajectory import (
        Trajectory,
        TrajectoryRecorder,
        save_trajectories,
    )
    from slidegen_rag.agent.tools.tools import (
        set_retrieval_augmentation,
        set_web_search_enabled,
    )
    from slidegen_rag.main import load_ra

    run_dir = run_dir or RunPaths.for_new_run(topic)
    generate_provider = "local" if lowcost else "gemini"

    question_file = runs_question_file(topic)
    reference_file = runs_reference_file(topic)
    index_path = _resolve_index_path(topic)

    questions = load_questions(question_file)
    references = load_references(reference_file)
    if not questions:
        raise ValueError(f"Không tìm thấy câu hỏi nào trong {question_file}")

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy index: {index_path}")

    questions_sha1 = run_config_mod.sha1_of_file_text(question_file)

    ra = load_ra(index_path)
    index_fp = run_config_mod.compute_index_fingerprint(ra, index_path)

    if multi_turn:
        set_retrieval_augmentation(ra)
        set_web_search_enabled(False)
        conversation_log = asyncio.run(
            _run_multi_turn_session(
                topic,
                questions,
                references,
                run_dir=run_dir,
                max_turns=max_turns,
                lowcost=lowcost,
                progress_callback=progress_callback,
            )
        )
        return AnswerBatch(
            results=conversation_log.pop("_results"),
            was_cached=False,
            source_run_id=None,
            generate_provider=generate_provider,
            index_fingerprint=index_fp,
            conversation_log=conversation_log,
        )

    target_fp = run_config_mod.config_fingerprint(
        {
            "mode": "full",
            "generate_provider": generate_provider,
            "gemini_model": _current_gemini_model(),
            "local_base_url": _current_local_base_url(),
            "index_source_sha1": index_fp["source_sha1"],
            "web_search_enabled": False,
            "max_turns": None,
            "questions_sha1": questions_sha1,
        }
    )

    if not force_rerun:

        def _cache_predicate(rp: RunPaths) -> bool:
            cfg = run_config_mod.load_run_config(rp)
            if cfg is None or cfg.get("mode") != "full":
                return False
            if cfg.get("config_fingerprint") != target_fp:
                return False
            cached = load_answers(rp.answer)
            return len(cached) >= len(questions) and all(
                q["num"] in cached for q in questions
            )

        source_run_id, reason = resolve_source_run(topic, predicate=_cache_predicate)
        if source_run_id is not None:
            source_rp = RunPaths.for_run(topic, source_run_id)
            logger.info(f"[{topic}] Cache hit ({reason})")
            if progress_callback:
                progress_callback(
                    f"[{topic}] ⚡ Dùng cache từ run {source_run_id} ({reason})"
                )
            for src_prop, dst_prop in (
                ("answer", "answer"),
                ("chunks", "chunks"),
                ("generate_state", "generate_state"),
                ("trajectory", "trajectory"),
            ):
                src = getattr(source_rp, src_prop)
                if src.exists():
                    dst = getattr(run_dir, dst_prop)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            cached_answers = load_answers(run_dir.answer)
            cached_chunks = load_chunks(run_dir.chunks)
            cached_trajectories = load_trajectory_texts(run_dir.trajectory)
            results = []
            for q in questions:
                num = q["num"]
                results.append(
                    {
                        "num": num,
                        "question": q["text"],
                        "response": cached_answers.get(num, ""),
                        "reference": references.get(num, ""),
                        "used_chunks": cached_chunks.get(num, []),
                        "trajectory": cached_trajectories.get(num),
                    }
                )
            return AnswerBatch(
                results=results,
                was_cached=True,
                source_run_id=source_run_id,
                generate_provider=generate_provider,
                index_fingerprint=index_fp,
            )
        else:
            logger.warning(f"[{topic}] Không dùng được cache: {reason}")
            if progress_callback:
                progress_callback(
                    f"[{topic}] ⚠️ Không dùng được cache ({reason}) — Đang khởi tạo CRAG Agent..."
                )
    elif progress_callback:
        progress_callback(f"[{topic}] Đang khởi tạo CRAG Agent...")

    set_retrieval_augmentation(ra)
    # Benchmark closed-book: reference.txt được viết trên giả định agent CHỈ
    # dùng data/src/pldc.md — tắt web search để không mâu thuẫn reference.
    set_web_search_enabled(False)
    set_generate_provider(generate_provider)
    graph = build_graph()
    recorder = TrajectoryRecorder(
        graph, tags=["evaluate", topic], metadata={"topic": topic}
    )

    results = []
    new_answers = {}
    new_chunks = {}
    new_generate_states: dict[int, dict] = {}
    new_trajectories: dict[int, Trajectory] = {}
    total = len(questions)

    gemini_running_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }

    for idx, q in enumerate(questions):
        num = q["num"]
        q_text = q["text"]
        ref = references.get(num, "")

        if progress_callback:
            progress_callback(
                f"[{topic}] Đang chạy Agent sinh câu trả lời {num}/{total}..."
            )

        logger.info(f"[{topic}] Agent đang trả lời câu {num}/{total}: {q_text[:70]}...")

        try:
            state_res, traj = recorder.run(
                make_initial_state(q_text),
                run_name=f"{topic}:Q{num}",
                metadata={"topic": topic, "question_num": num},
            )
            answer = (state_res.get("generation") or "").strip()
            used_docs = state_res.get("documents", [])
            generate_state = {
                "facet_sufficiency": state_res.get("facet_sufficiency") or [],
                "document_priorities": state_res.get("document_priorities") or [],
            }
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi chạy Agent câu {num}: {e}")
            answer = f"[LỖI AGENT] {e}"
            used_docs = []
            generate_state = {"facet_sufficiency": [], "document_priorities": []}
            traj = Trajectory(query=q_text, started_at=datetime.now(), error=str(e))

        new_answers[num] = answer
        new_chunks[num] = used_docs
        new_generate_states[num] = generate_state
        new_trajectories[num] = traj

        q_usage = traj.token_usage or {}
        q_gemini_tokens = q_usage.get(
            "gemini_total_tokens", q_usage.get("total_tokens", 0)
        )
        q_cost_usd = q_usage.get("total_cost_usd", 0.0)
        gemini_running_total["input_tokens"] += q_usage.get(
            "gemini_input_tokens", q_usage.get("input_tokens", 0)
        )
        gemini_running_total["output_tokens"] += q_usage.get(
            "gemini_output_tokens", q_usage.get("output_tokens", 0)
        )
        gemini_running_total["total_tokens"] += q_gemini_tokens
        gemini_running_total["cost_usd"] += q_cost_usd

        usage_msg = (
            f"[{topic}] Câu {num}/{total}: Gemini +{q_gemini_tokens:,} token (~${q_cost_usd:.4f}) "
            f"| Tích lũy: {gemini_running_total['total_tokens']:,} token "
            f"(~${gemini_running_total['cost_usd']:.4f} USD)"
        )
        logger.info(usage_msg)
        if progress_callback:
            progress_callback(usage_msg)

        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": ref,
                "used_chunks": used_docs,
                "trajectory": traj.render_text(),
                "retrieve_count": traj.retrieve_count,
                "tool_calls_total": sum(len(s.tool_calls) for s in traj.steps),
                "llm_calls_total": q_usage.get("calls", 0),
                "agent_latency_s": traj.duration_s,
                "token_usage": q_usage,
            }
        )

    save_answers(run_dir.answer, new_answers)
    save_chunks(run_dir.chunks, new_chunks)
    save_generate_state(run_dir.generate_state, new_generate_states)
    save_trajectories(run_dir.trajectory, new_trajectories, topic)
    logger.info(
        f"[{topic}] Đã lưu câu trả lời/chunks/generate_state/trajectory vào {run_dir.root}"
    )
    logger.info(
        f"[{topic}] Tổng Gemini token cả lần chạy: "
        f"in={gemini_running_total['input_tokens']:,} out={gemini_running_total['output_tokens']:,} "
        f"tổng={gemini_running_total['total_tokens']:,} "
        f"| Chi phí ước tính: ${gemini_running_total['cost_usd']:.4f} USD "
        f"(~{gemini_running_total['cost_usd'] * 25400:.0f} VNĐ)"
    )

    return AnswerBatch(
        results=results,
        was_cached=False,
        source_run_id=None,
        generate_provider=generate_provider,
        index_fingerprint=index_fp,
    )


async def _run_multi_turn_session(
    topic: str,
    questions: list[dict],
    references: dict[int, str],
    *,
    run_dir: RunPaths,
    max_turns: int,
    lowcost: bool,
    progress_callback=None,
) -> dict:
    """
    Chạy 30 câu hỏi benchmark như MỘT phiên hội thoại liền mạch có short-term
    memory (`ConversationManager`, repo=None → thuần in-memory) — xem
    plan5.md Phần B. Bám sát trình tự `main.py::interactive_loop`
    (contextualize -> agent -> record_turn -> maybe_compact), khác main.py ở
    chỗ dùng MỘT `TrajectoryRecorder.arun()` cho cả 30 lượt qua `asyncio.run`
    duy nhất ở lớp gọi ngoài (`get_answers_and_chunks_for_topic`), chứ không
    `recorder.run()` (mở event loop MỚI mỗi câu — sẽ làm hỏng
    `_contextualize_llm`/`_compact_llm` module-level dùng chung).

    Returns:
        dict — `conversation.json` schema (xem plan5.md D3) CỘNG THÊM khoá
        nội bộ `_results` (list[dict], cùng shape các nhánh khác của
        `get_answers_and_chunks_for_topic`) — `get_answers_and_chunks_for_topic`
        pop() khoá này ra trước khi trả `AnswerBatch.conversation_log`.
    """
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.graph.conversation import ConversationManager
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.models.llm import LLMModel, TokenTracker
    from slidegen_rag.agent.nodes.generate_node import set_generate_provider
    from slidegen_rag.agent.observability.trajectory import (
        Trajectory,
        TrajectoryRecorder,
        save_trajectories,
    )

    set_generate_provider("local" if lowcost else "gemini")
    graph = build_graph()
    recorder = TrajectoryRecorder(
        graph, tags=["evaluate", "multiturn", topic], metadata={"topic": topic}
    )
    conv = ConversationManager(max_turns=max_turns)

    results: list[dict] = []
    new_answers: dict[int, str] = {}
    new_chunks: dict[int, list[str]] = {}
    new_generate_states: dict[int, dict] = {}
    new_trajectories: dict[int, Trajectory] = {}
    turns_log: list[dict] = []
    compactions_log: list[dict] = []

    total = len(questions)
    for idx, q in enumerate(questions, 1):
        num = q["num"]
        q_text = q["text"]
        ref = references.get(num, "")

        if progress_callback:
            progress_callback(
                f"[{topic}] [multi-turn] Lượt {idx}/{total} (câu {num}): {q_text[:60]}..."
            )

        history_entries_before = len(conv.chat_history)

        t_ctx0 = time.perf_counter()
        u_ctx0 = LLMModel.tracker.snapshot()
        standalone = await conv.contextualize(q_text)
        mem_ctx_usage = TokenTracker.summary_from_delta(
            u_ctx0, LLMModel.tracker.snapshot()
        )
        t_ctx = time.perf_counter() - t_ctx0
        was_rewritten = standalone.strip() != q_text.strip()
        seeds = conv.seed_drill_candidates()

        turn_degraded = False
        try:
            state_res, traj = await recorder.arun(
                make_initial_state(standalone, drill_candidates=seeds),
                run_name=f"{topic}:T{idx}:Q{num}",
                metadata={"topic": topic, "question_num": num, "turn": idx},
            )
            answer = (state_res.get("generation") or "").strip()
            used_docs = state_res.get("documents", [])
            document_node_ids = state_res.get("document_node_ids", [])
            generate_state = {
                "facet_sufficiency": state_res.get("facet_sufficiency") or [],
                "document_priorities": state_res.get("document_priorities") or [],
            }
        except Exception as e:
            logger.error(f"[{topic}] [multi-turn] Lỗi lượt {idx} (câu {num}): {e}")
            answer = f"[LỖI AGENT] {e}"
            used_docs = []
            document_node_ids = []
            generate_state = {"facet_sufficiency": [], "document_priorities": []}
            traj = Trajectory(query=standalone, started_at=datetime.now(), error=str(e))
            turn_degraded = True

        # record_turn ghi q_text GỐC (không phải standalone) — xem plan5.md
        # B1: lịch sử phải đúng như người dùng hỏi. Lượt lỗi ghi placeholder
        # (không nhét traceback vào prompt của 29 lượt sau / mọi lần nén) và
        # [] node ids (chặn citation rác lan sang lượt kế) nhưng VẪN
        # record_turn — bỏ hẳn sẽ làm "30 lượt" âm thầm thành ít hơn 30 và
        # mọi so sánh overhead lệch số câu.
        if turn_degraded:
            conv.record_turn(q_text, "(hệ thống không trả lời được câu này)", [])
        else:
            conv.record_turn(q_text, answer, document_node_ids)

        history_before_compact = len(conv.chat_history)
        t_cmp0 = time.perf_counter()
        u_cmp0 = LLMModel.tracker.snapshot()
        await conv.maybe_compact()
        mem_cmp_usage = TokenTracker.summary_from_delta(
            u_cmp0, LLMModel.tracker.snapshot()
        )
        t_cmp = time.perf_counter() - t_cmp0
        compacted_this_turn = len(conv.chat_history) < history_before_compact

        mem_usage = _add_token_summaries(mem_ctx_usage, mem_cmp_usage)
        memory_latency_s = t_ctx + t_cmp
        agent_latency_s = traj.duration_s
        turn_latency_s = memory_latency_s + agent_latency_s

        new_answers[num] = answer
        new_chunks[num] = used_docs
        new_generate_states[num] = generate_state
        new_trajectories[num] = traj

        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": ref,
                "used_chunks": used_docs,
                "trajectory": traj.render_text(),
                "mode": "multi_turn",
                "turn_index": idx,
                "standalone_query": standalone,
                "was_rewritten": was_rewritten,
                "seed_candidates_count": len(seeds),
                "history_entries_before": history_entries_before,
                "compacted_this_turn": compacted_this_turn,
                "turn_degraded": turn_degraded,
                "retrieve_count": traj.retrieve_count,
                "tool_calls_total": sum(len(s.tool_calls) for s in traj.steps),
                "llm_calls_total": (traj.token_usage or {}).get("calls", 0),
                "agent_latency_s": agent_latency_s,
                "memory_latency_s": memory_latency_s,
                "turn_latency_s": turn_latency_s,
                "token_usage": traj.token_usage or {},
                "mem_usage": mem_usage,
            }
        )

        turns_log.append(
            {
                "turn": idx,
                "question_num": num,
                "raw_query": q_text,
                "standalone_query": standalone,
                "was_rewritten": was_rewritten,
                "seed_drill_candidates": seeds,
                "cited_node_ids": [] if turn_degraded else document_node_ids,
                "history_entries_before": history_entries_before,
                "history_entries_after": len(conv.chat_history),
                "compacted": compacted_this_turn,
                "turn_degraded": turn_degraded,
            }
        )
        if compacted_this_turn:
            compactions_log.append(
                {
                    "after_turn": idx,
                    "entries_folded": history_before_compact - len(conv.chat_history),
                    "summary": conv.history_summary,
                    "duration_s": t_cmp,
                    "gemini_tokens": mem_cmp_usage.get("gemini_total_tokens", 0),
                }
            )

    save_answers(run_dir.answer, new_answers)
    save_chunks(run_dir.chunks, new_chunks)
    save_generate_state(run_dir.generate_state, new_generate_states)
    save_trajectories(run_dir.trajectory, new_trajectories, topic)

    return {
        "topic": topic,
        "mode": "multiturn",
        "max_turns": max_turns,
        "turns": turns_log,
        "compactions": compactions_log,
        "final_history_summary": conv.history_summary,
        "_results": results,
    }


def regenerate_answers_for_topic(
    topic: str,
    progress_callback=None,
    lowcost: bool = False,
    run_dir: Optional[RunPaths] = None,
    source_run_id: Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Chạy lại DUY NHẤT generate_node trên bộ chunk đã có sẵn ở một run NGUỒN
    (xem `resolve_source_run`), KHÔNG chạy retrieve/grade/context_evaluator
    và không cần nạp index.

    Ghi answer.txt MỚI vào `run_dir` (run mới, tự chứa) nhưng chunks.txt/
    generate_state.json/trajectory.txt được COPY nguyên vẹn từ run nguồn —
    chunk là đầu vào cố định, trajectory cũ vẫn mô tả đúng cách bộ chunk đó
    được tìm ra.

    Args:
        lowcost: True → generate_node dùng model tự host thay Gemini.
        source_run_id: chỉ định tường minh run nguồn (vd người dùng chọn
            trên UI) — mặc định tự chọn qua `resolve_source_run`.

    Returns:
        (results, source_run_id) — results cùng shape với
        `AnswerBatch.results` để `run_evaluation_for_topic` dùng lại nguyên
        si đường chấm điểm.
    """
    from slidegen_rag.agent.models.llm import LLMModel, TokenTracker
    from slidegen_rag.agent.nodes.generate_node import (
        generate_node,
        set_generate_provider,
    )

    run_dir = run_dir or RunPaths.for_new_run(topic)

    def _predicate(rp: RunPaths) -> bool:
        if not rp.generate_state.exists():
            return False
        cached_chunks = load_chunks(rp.chunks)
        return any(cached_chunks.values())

    resolved_id, reason = resolve_source_run(
        topic, explicit_run_id=source_run_id, predicate=_predicate
    )
    if resolved_id is None:
        raise FileNotFoundError(
            f"[{topic}] Không tìm được run nguồn có sẵn {GENERATE_STATE_FILENAME} và "
            f"chunks.txt khác rỗng cho chế độ 'chỉ chạy lại generate' ({reason}). "
            f"Hãy chạy đầy đủ 1 lần trước."
        )
    source_rp = RunPaths.for_run(topic, resolved_id)

    question_file = runs_question_file(topic)
    reference_file = runs_reference_file(topic)

    questions = load_questions(question_file)
    references = load_references(reference_file)
    cached_chunks = load_chunks(source_rp.chunks)
    cached_states = load_generate_state(source_rp.generate_state)
    cached_trajectories = load_trajectory_texts(source_rp.trajectory)

    if not questions:
        raise ValueError(f"Không tìm thấy câu hỏi nào trong {question_file}")

    missing = [q["num"] for q in questions if not cached_chunks.get(q["num"])]
    if missing:
        logger.warning(
            f"[{topic}] {len(missing)} câu không có chunk trong run nguồn {resolved_id} "
            f"({missing[:10]}) — vẫn chạy generate với ngữ cảnh rỗng cho các câu này"
        )

    # Copy chunks/generate_state/trajectory NGUYÊN VẸN — run mới vẫn tự chứa
    # (xem C1) dù không sinh lại chunk.
    for prop in ("chunks", "generate_state", "trajectory"):
        src = getattr(source_rp, prop)
        if src.exists():
            dst = getattr(run_dir, prop)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    set_generate_provider("local" if lowcost else "gemini")
    LLMModel.reset_usage()

    results = []
    new_answers = {}
    total = len(questions)

    for idx, q in enumerate(questions, 1):
        num = q["num"]
        q_text = q["text"]
        docs = cached_chunks.get(num, [])
        state = cached_states.get(num, {})
        priorities = state.get("document_priorities") or []

        if progress_callback:
            progress_callback(
                f"[{topic}] Đang sinh lại câu trả lời {idx}/{total} "
                f"({len(docs)} chunk có sẵn)..."
            )

        t0 = time.perf_counter()
        u0 = LLMModel.tracker.snapshot()
        try:
            out = asyncio.run(
                generate_node(
                    {
                        "query": q_text,
                        "documents": docs,
                        "document_priorities": priorities,
                        "facet_sufficiency": state.get("facet_sufficiency") or [],
                    }
                )
            )
            answer = (out.get("generation") or "").strip()
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi sinh lại câu {num}: {e}")
            answer = f"[LỖI GENERATE] {e}"
        latency_s = time.perf_counter() - t0
        q_token_usage = TokenTracker.summary_from_delta(u0, LLMModel.tracker.snapshot())

        if priorities and len(priorities) != len(docs):
            logger.warning(
                f"[{topic}] Câu {num}: document_priorities ({len(priorities)}) lệch "
                f"số chunk ({len(docs)}) — build_context bỏ qua priority, cắt theo "
                f"thứ tự chunk thay vì theo chất lượng"
            )

        new_answers[num] = answer
        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": references.get(num, ""),
                "used_chunks": docs,
                "trajectory": cached_trajectories.get(num),
                "agent_latency_s": latency_s,
                "token_usage": q_token_usage,
                "retrieve_count": None,
                "tool_calls_total": 0,
                "llm_calls_total": q_token_usage.get("calls", 0),
            }
        )

    save_answers(run_dir.answer, new_answers)

    usage = LLMModel.get_usage_summary()
    cost = usage.get("total_cost_usd", 0.0)
    logger.info(
        f"[{topic}] Đã sinh lại {total} câu trả lời (chỉ generate, nguồn={resolved_id}) "
        f"và ghi vào {run_dir.answer} | Gemini token: "
        f"in={usage.get('gemini_input_tokens', 0):,} out={usage.get('gemini_output_tokens', 0):,} "
        f"| Chi phí ước tính: ${cost:.4f} USD (~{cost * 25400:.0f} VNĐ)"
    )

    return results, resolved_id


def _parse_trajectory_summary(
    traj_text: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Trích 3 chỉ số ngắn gọn từ text trajectory (dùng chung cho cả câu vừa
    chạy lẫn câu nạp từ cache — cả hai đều ở dạng text render sẵn, nên chỉ
    cần parse text thay vì phải giữ lại Trajectory object).

    Returns:
        (traj_loops: "2/3" hoặc None,
         traj_tools: "semantic_search x1, ..." hoặc None,
         traj_facet_loops: "label1=n1, label2=n2, ..." hoặc None)
    """
    if not traj_text:
        return None, None, None
    loop_match = re.search(r"Vòng lặp retrieve:\s*(\d+/\d+)", traj_text)
    tools_match = re.search(r"Tool đã gọi\s*:\s*(.+)", traj_text)
    facet_loops_match = re.search(r"Vòng lặp theo facet:\s*(.+)", traj_text)
    traj_loops = loop_match.group(1) if loop_match else None
    traj_tools = tools_match.group(1).strip() if tools_match else None
    traj_facet_loops = facet_loops_match.group(1).strip() if facet_loops_match else None
    return traj_loops, traj_tools, traj_facet_loops


def run_evaluation_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
    lowcost: bool = False,
    regenerate_only: bool = False,
    multi_turn: bool = False,
    max_turns: int = 5,
) -> tuple[pd.DataFrame, dict, bool, str]:
    """
    Thực hiện đánh giá toàn diện một chủ đề — tạo MỘT run mới tự chứa trong
    `runs/<run_id>/` (xem plan5.md Phần C), rồi chấm điểm. Nếu một câu hỏi
    bị lỗi, câu đó được đánh dấu FAILED và KHÔNG TÍNH VÀO ĐIỂM TỔNG.

    Args:
        lowcost: True → generate_node dùng model tự host thay Gemini.
        regenerate_only: True → không chạy agent, chỉ chạy lại generate_node
            trên chunk có sẵn ở run nguồn rồi chấm.
        multi_turn: True → 30 câu chạy như MỘT phiên hội thoại liền mạch có
            short-term memory (xem plan5.md Phần B). Bỏ qua `force_rerun` —
            luôn chạy phiên mới.
        max_turns: cửa sổ lịch sử của `ConversationManager` khi multi_turn.

    Returns:
        (detail_df, summary_dict, was_cached, run_id)
    """
    from slidegen_rag.agent.models.llm import TokenTracker

    started_at = datetime.now()
    run_dir = RunPaths.for_new_run(topic, now=started_at)
    session_t0 = time.perf_counter()

    if regenerate_only:
        results, source_run_id = regenerate_answers_for_topic(
            topic, progress_callback=progress_callback, lowcost=lowcost, run_dir=run_dir
        )
        was_cached = False
        run_mode = "regenerate_only"
        generate_provider = "local" if lowcost else "gemini"
        conversation_log = None
        source_cfg = None
        if source_run_id:
            source_cfg = run_config_mod.load_run_config(
                RunPaths.for_run(topic, source_run_id)
            )
        index_fp = (
            (source_cfg or {}).get("index")
            or {"path": None, "source_sha1": None, "pkl_sha1": None}
        )
    else:
        batch = get_answers_and_chunks_for_topic(
            topic,
            force_rerun=force_rerun,
            progress_callback=progress_callback,
            lowcost=lowcost,
            multi_turn=multi_turn,
            max_turns=max_turns,
            run_dir=run_dir,
        )
        results = batch.results
        was_cached = batch.was_cached
        source_run_id = batch.source_run_id
        generate_provider = batch.generate_provider
        conversation_log = batch.conversation_log
        index_fp = batch.index_fingerprint
        run_mode = "cached" if was_cached else ("multiturn" if multi_turn else "full")

    primary_llm, fallback_llm = create_evaluator_llm()
    if not primary_llm and not fallback_llm:
        raise RuntimeError(
            "Không có LLM Judge nào khả dụng. Vui lòng kiểm tra LLM_BASE_URL hoặc GEMINI_API_KEY trong .env"
        )

    rows = []
    total = len(results)

    for idx, item in enumerate(results, 1):
        num = item["num"]
        q_text = item["question"]
        resp = item["response"]
        ref = item["reference"]
        chunks = item["used_chunks"]
        traj_text = item.get("trajectory")
        traj_loops, traj_tools, traj_facet_loops = _parse_trajectory_summary(traj_text)

        if progress_callback:
            progress_callback(f"[{topic}] LLM Judge đang chấm câu {idx}/{total}...")

        logger.info(f"[{topic}] Đang chấm câu {idx}/{total}...")

        t_judge0 = time.perf_counter()
        u_judge0 = judge_tracker.snapshot()
        eval_res = asyncio.run(
            evaluate_single_answer_with_judge(
                primary_llm, fallback_llm, q_text, ref, resp, chunks
            )
        )
        judge_latency_s = time.perf_counter() - t_judge0
        judge_usage = TokenTracker.summary_from_delta(
            u_judge0, judge_tracker.snapshot()
        )

        coverage = calculate_keyword_coverage(ref, chunks)
        coverage_strict = calculate_keyword_coverage_strict(ref, chunks)
        answer_coverage = calculate_answer_coverage(ref, resp)
        utilization = (
            round(answer_coverage / coverage_strict * 100.0, 1)
            if coverage_strict > 0
            else 0.0
        )

        if eval_res is not None:
            is_success = True
            acc_val = round(eval_res.accuracy, 2)
            comp_val = round(eval_res.completeness, 2)
            rel_val = round(eval_res.relevance, 2)
            faith_val = round(eval_res.faithfulness, 2)
            total_score = round((acc_val + comp_val + rel_val + faith_val) / 4.0, 2)
            feedback_val = eval_res.feedback
        else:
            is_success = False
            acc_val = comp_val = rel_val = faith_val = total_score = None
            feedback_val = (
                "[LỖI ĐÁNH GIÁ] Không nhận được phản hồi hợp lệ từ LLM Judge."
            )

        token_usage = item.get("token_usage") or _EMPTY_TOKEN_SUMMARY
        mem_usage = item.get("mem_usage") or _EMPTY_TOKEN_SUMMARY
        agent_latency_s = item.get("agent_latency_s")
        memory_latency_s = item.get("memory_latency_s", 0.0) or 0.0
        turn_latency_s = item.get(
            "turn_latency_s",
            (agent_latency_s or 0.0) + memory_latency_s,
        )

        rows.append(
            {
                "STT": num,
                "topic": topic,
                "question": q_text,
                "response": resp,
                "reference": ref,
                "accuracy": acc_val,
                "completeness": comp_val,
                "relevance": rel_val,
                "faithfulness": faith_val,
                "total_score": total_score,
                "coverage_pct": round(coverage, 1),
                "coverage_pct_strict": round(coverage_strict, 1),
                "answer_coverage_pct": round(answer_coverage, 1),
                "utilization_pct": utilization,
                "chunks_count": len(chunks),
                "feedback": feedback_val,
                "is_success": is_success,
                "used_chunks": chunks,
                "traj_loops": traj_loops,
                "traj_tools": traj_tools,
                "traj_facet_loops": traj_facet_loops,
                "traj_text": traj_text
                or "(chưa có trajectory — cache cũ trước khi bật tính năng này)",
                # ── Cột mới (xem plan5.md B4) ──────────────────────────
                "mode": run_mode,
                "turn_index": item.get("turn_index", num),
                "standalone_query": item.get("standalone_query", q_text),
                "was_rewritten": item.get("was_rewritten", False),
                "seed_candidates_count": item.get("seed_candidates_count", 0),
                "history_entries_before": item.get("history_entries_before", 0),
                "compacted_this_turn": item.get("compacted_this_turn", False),
                "turn_degraded": item.get("turn_degraded", False),
                "retrieve_count": item.get("retrieve_count"),
                "tool_calls_total": item.get("tool_calls_total"),
                "llm_calls_total": item.get("llm_calls_total"),
                "agent_latency_s": agent_latency_s,
                "memory_latency_s": memory_latency_s,
                "turn_latency_s": turn_latency_s,
                "judge_latency_s": judge_latency_s,
                "agent_gemini_input_tokens": token_usage.get("gemini_input_tokens", 0),
                "agent_gemini_output_tokens": token_usage.get(
                    "gemini_output_tokens", 0
                ),
                "agent_gemini_total_tokens": token_usage.get("gemini_total_tokens", 0),
                "agent_gemini_calls": token_usage.get("gemini_calls", 0),
                "agent_local_tokens": token_usage.get("local_total_tokens", 0),
                "agent_local_calls": token_usage.get("local_calls", 0),
                "mem_gemini_input_tokens": mem_usage.get("gemini_input_tokens", 0),
                "mem_gemini_output_tokens": mem_usage.get("gemini_output_tokens", 0),
                "mem_gemini_total_tokens": mem_usage.get("gemini_total_tokens", 0),
                "mem_gemini_calls": mem_usage.get("gemini_calls", 0),
                "judge_gemini_tokens": judge_usage.get("gemini_total_tokens", 0),
                "judge_local_tokens": judge_usage.get("local_total_tokens", 0),
                "judge_calls": judge_usage.get("calls", 0),
                "agent_cost_usd": token_usage.get("total_cost_usd", 0.0),
                "mem_cost_usd": mem_usage.get("total_cost_usd", 0.0),
                "turn_cost_usd": token_usage.get("total_cost_usd", 0.0)
                + mem_usage.get("total_cost_usd", 0.0),
                "judge_cost_usd": judge_usage.get("total_cost_usd", 0.0),
            }
        )

    detail_df = pd.DataFrame(rows)

    valid_df = detail_df[detail_df["is_success"] == True]  # noqa: E712
    success_count = len(valid_df)
    failed_count = len(detail_df) - success_count

    if not valid_df.empty:
        summary = {
            "accuracy": float(valid_df["accuracy"].astype(float).mean()),
            "completeness": float(valid_df["completeness"].astype(float).mean()),
            "relevance": float(valid_df["relevance"].astype(float).mean()),
            "faithfulness": float(valid_df["faithfulness"].astype(float).mean()),
            "total_score": float(valid_df["total_score"].astype(float).mean()),
            "coverage_pct": float(valid_df["coverage_pct"].astype(float).mean()),
            "coverage_pct_strict": float(
                valid_df["coverage_pct_strict"].astype(float).mean()
            ),
            "answer_coverage_pct": float(
                valid_df["answer_coverage_pct"].astype(float).mean()
            ),
            "utilization_pct": float(valid_df["utilization_pct"].astype(float).mean()),
            "avg_chunks": float(detail_df["chunks_count"].mean()),
            "total_questions": len(detail_df),
            "success_count": success_count,
            "failed_count": failed_count,
            "generate_provider": generate_provider,
            "regenerate_only": regenerate_only,
        }
    else:
        summary = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "faithfulness": 0.0,
            "total_score": 0.0,
            "coverage_pct": 0.0,
            "coverage_pct_strict": 0.0,
            "answer_coverage_pct": 0.0,
            "utilization_pct": 0.0,
            "avg_chunks": 0.0,
            "total_questions": len(detail_df),
            "success_count": 0,
            "failed_count": failed_count,
            "generate_provider": generate_provider,
            "regenerate_only": regenerate_only,
        }

    finished_at = datetime.now()
    session_wallclock_s = time.perf_counter() - session_t0

    compaction_count = int(detail_df["compacted_this_turn"].sum())
    rewritten_count = int(detail_df["was_rewritten"].sum())
    degraded_turns = int(detail_df["turn_degraded"].sum())
    contextualize_count = int((detail_df["mem_gemini_calls"] > 0).sum())

    summary.update(
        {
            "mode": run_mode,
            "run_id": run_dir.run_id,
            "max_turns": max_turns if multi_turn else None,
            "compaction_count": compaction_count,
            "contextualize_count": contextualize_count,
            "rewritten_count": rewritten_count,
            "degraded_turns": degraded_turns,
            "session_wallclock_s": session_wallclock_s,
            "total_agent_latency_s": float(detail_df["agent_latency_s"].dropna().sum())
            if detail_df["agent_latency_s"].notna().any()
            else None,
            "total_memory_latency_s": float(detail_df["memory_latency_s"].sum()),
            "total_judge_latency_s": float(detail_df["judge_latency_s"].sum()),
            "avg_turn_latency_s": float(detail_df["turn_latency_s"].dropna().mean())
            if detail_df["turn_latency_s"].notna().any()
            else None,
            "total_agent_gemini_tokens": int(
                detail_df["agent_gemini_total_tokens"].sum()
            ),
            "total_agent_gemini_input_tokens": int(
                detail_df["agent_gemini_input_tokens"].sum()
            ),
            "total_agent_gemini_output_tokens": int(
                detail_df["agent_gemini_output_tokens"].sum()
            ),
            "total_agent_local_tokens": int(detail_df["agent_local_tokens"].sum()),
            "total_mem_gemini_tokens": int(detail_df["mem_gemini_total_tokens"].sum()),
            "total_judge_tokens": int(
                detail_df["judge_gemini_tokens"].sum()
                + detail_df["judge_local_tokens"].sum()
            ),
            "total_gemini_calls": int(
                detail_df["agent_gemini_calls"].sum()
                + detail_df["mem_gemini_calls"].sum()
            ),
            "total_local_calls": int(detail_df["agent_local_calls"].sum()),
            "avg_retrieve_count": float(detail_df["retrieve_count"].dropna().mean())
            if detail_df["retrieve_count"].notna().any()
            else None,
            "avg_tool_calls": float(detail_df["tool_calls_total"].dropna().mean())
            if detail_df["tool_calls_total"].notna().any()
            else None,
            # Chi phí mỗi câu KHÔNG cộng judge (xem plan5.md A2) — judge báo
            # cáo riêng ở total_judge_tokens/judge_cost_usd.
            "total_cost_usd": float(detail_df["turn_cost_usd"].sum()),
            "total_cost_vnd": float(detail_df["turn_cost_usd"].sum()) * 25400,
            "mem_cost_usd": float(detail_df["mem_cost_usd"].sum()),
            "judge_cost_usd": float(detail_df["judge_cost_usd"].sum()),
            "cost_excludes_judge": True,
            "pricing_model": _current_gemini_model(),
            "usd_vnd_rate": 25400,
        }
    )

    cfg = run_config_mod.build_run_config(
        run_paths=run_dir,
        topic=topic,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        mode=run_mode,
        generate_provider=generate_provider,
        lowcost=lowcost,
        max_turns=max_turns if multi_turn else None,
        web_search_enabled=False,
        max_retrieve=_agent_max_retrieve(),
        gemini_model=_current_gemini_model(),
        local_base_url=_current_local_base_url(),
        index_path=Path(index_fp["path"]) if index_fp.get("path") else _resolve_index_path(topic),
        index_source_sha1=index_fp.get("source_sha1"),
        index_pkl_sha1=index_fp.get("pkl_sha1"),
        questions_sha1=run_config_mod.sha1_of_file_text(
            runs_question_file(topic)
        ),
        references_sha1=run_config_mod.sha1_of_file_text(
            runs_reference_file(topic)
        ),
        total_questions=len(detail_df),
        source_run_id=source_run_id,
        migrated=False,
    )
    run_config_mod.save_run_config(run_dir, cfg)

    from slidegen_rag.evaluate.report import export_evaluation_results

    export_evaluation_results(run_dir, detail_df, summary)
    write_latest_pointer(topic, run_dir.run_id)

    if conversation_log is not None:
        import json

        with open(run_dir.conversation, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, ensure_ascii=False, indent=2)
        turn_metrics = [
            {
                "STT": r["STT"],
                "turn_index": r["turn_index"],
                "agent_latency_s": r["agent_latency_s"],
                "memory_latency_s": r["memory_latency_s"],
                "judge_latency_s": r["judge_latency_s"],
                "raw_usage": {
                    "agent": next(
                        (
                            item.get("token_usage")
                            for item in results
                            if item["num"] == r["STT"]
                        ),
                        None,
                    ),
                    "memory": next(
                        (
                            item.get("mem_usage")
                            for item in results
                            if item["num"] == r["STT"]
                        ),
                        None,
                    ),
                },
            }
            for r in rows
        ]
        with open(run_dir.turn_metrics, "w", encoding="utf-8") as f:
            json.dump(turn_metrics, f, ensure_ascii=False, indent=2)

    return detail_df, summary, was_cached, run_dir.run_id


def _agent_max_retrieve() -> int:
    from slidegen_rag.config import AgentConfig

    return AgentConfig.MAX_RETRIEVE
