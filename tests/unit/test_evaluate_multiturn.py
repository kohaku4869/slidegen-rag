"""`_run_multi_turn_session` (evaluate/runner.py) với graph GIẢ — xem
plan5.md Phần B, bước verification #9.

Không dựng graph/index thật: monkeypatch `TrajectoryRecorder` (import cục bộ
trong `_run_multi_turn_session`) bằng một fake trả về (state, Trajectory) đã
biết trước, và monkeypatch 2 LLM module-level của `ConversationManager`
(`_contextualize_llm`/`_compact_llm`) để không gọi mạng thật."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from slidegen_rag.agent.graph import conversation as conv_mod
from slidegen_rag.agent.observability.trajectory import Trajectory
from slidegen_rag.evaluate import runner
from slidegen_rag.evaluate.runs import RunPaths

TOTAL_TURNS = 30
FAIL_TURN = 13  # 1-indexed turn cố ý lỗi để test nhánh turn_degraded


def _questions():
    return [{"num": i, "text": f"Câu hỏi số {i}?"} for i in range(1, TOTAL_TURNS + 1)]


def _references():
    return {i: f"Đáp án {i}" for i in range(1, TOTAL_TURNS + 1)}


class _FakeRecorder:
    """Thay `TrajectoryRecorder` thật — không chạm LangGraph/LLM."""

    def __init__(self, graph, *, run_name_prefix="crag", tags=None, metadata=None, on_step=None):
        pass

    async def arun(self, initial_state, *, run_name=None, tags=None, metadata=None):
        # run_name = f"{topic}:T{idx}:Q{num}" (xem runner.py::_run_multi_turn_session)
        idx = int(run_name.split(":T")[1].split(":Q")[0])
        if idx == FAIL_TURN:
            raise RuntimeError("agent lỗi giả lập")
        state_res = {
            "generation": f"Trả lời {idx}",
            "documents": [f"chunk-{idx}"],
            "document_node_ids": [idx],
            "facet_sufficiency": [],
            "document_priorities": [],
        }
        traj = Trajectory(
            query=initial_state["query"],
            started_at=datetime.now(),
            duration_s=0.01,
            retrieve_count=1,
        )
        return state_res, traj


@pytest.fixture
def _patched_multiturn(monkeypatch):
    monkeypatch.setattr(
        "slidegen_rag.agent.graph.builder.build_graph", lambda: object()
    )
    monkeypatch.setattr(
        "slidegen_rag.agent.observability.trajectory.TrajectoryRecorder",
        _FakeRecorder,
    )
    monkeypatch.setattr(
        "slidegen_rag.agent.nodes.generate_node.set_generate_provider",
        lambda provider: None,
    )

    ctx_calls: list = []
    cmp_calls: list = []

    async def _fake_ctx_ainvoke(messages):
        ctx_calls.append(messages)
        return SimpleNamespace(content="Câu hỏi đã viết lại")

    async def _fake_cmp_ainvoke(messages):
        cmp_calls.append(messages)
        return SimpleNamespace(content="Tóm tắt lịch sử cũ")

    monkeypatch.setattr(conv_mod._contextualize_llm, "ainvoke", _fake_ctx_ainvoke)
    monkeypatch.setattr(conv_mod._compact_llm, "ainvoke", _fake_cmp_ainvoke)
    return ctx_calls, cmp_calls


def test_first_turn_skips_contextualize(tmp_path, monkeypatch, _patched_multiturn):
    from slidegen_rag.evaluate import runs as runs_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    ctx_calls, _cmp_calls = _patched_multiturn
    rp = RunPaths.for_new_run("chu_de_test")

    import asyncio

    log = asyncio.run(
        runner._run_multi_turn_session(
            "chu_de_test",
            _questions(),
            _references(),
            run_dir=rp,
            max_turns=5,
            lowcost=False,
            progress_callback=None,
        )
    )

    turn1 = log["_results"][0]
    assert turn1["turn_index"] == 1
    assert turn1["was_rewritten"] is False
    assert turn1["history_entries_before"] == 0
    # 29 lượt còn lại (2->30) mới gọi contextualize -> đúng 29 lần.
    assert len(ctx_calls) == TOTAL_TURNS - 1


def test_compaction_count_is_25_for_30_turns(tmp_path, monkeypatch, _patched_multiturn):
    from slidegen_rag.evaluate import runs as runs_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    import asyncio

    log = asyncio.run(
        runner._run_multi_turn_session(
            "chu_de_test",
            _questions(),
            _references(),
            run_dir=rp,
            max_turns=5,
            lowcost=False,
            progress_callback=None,
        )
    )

    assert len(log["compactions"]) == 25
    compacted_flags = [t["compacted"] for t in log["turns"]]
    assert compacted_flags.count(True) == 25
    # Nén bắt đầu từ lượt 6 (max_entries=10, mỗi lượt +2 entry).
    assert compacted_flags[:5] == [False] * 5
    assert all(compacted_flags[5:])


def test_failing_turn_writes_placeholder_not_raw_error_into_history(
    tmp_path, monkeypatch, _patched_multiturn
):
    from slidegen_rag.evaluate import runs as runs_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    ctx_calls, _cmp_calls = _patched_multiturn
    rp = RunPaths.for_new_run("chu_de_test")

    import asyncio

    log = asyncio.run(
        runner._run_multi_turn_session(
            "chu_de_test",
            _questions(),
            _references(),
            run_dir=rp,
            max_turns=5,
            lowcost=False,
            progress_callback=None,
        )
    )

    results = log["_results"]
    failed = results[FAIL_TURN - 1]
    assert failed["turn_degraded"] is True
    assert failed["response"].startswith("[LỖI AGENT]")

    # turn_degraded vẫn record_turn (giữ đúng 1 lượt <-> 1 câu hỏi) — không
    # bị bỏ hẳn khỏi turns log.
    turns_log = log["turns"]
    assert len(turns_log) == TOTAL_TURNS
    assert turns_log[FAIL_TURN - 1]["turn_degraded"] is True
    assert turns_log[FAIL_TURN - 1]["cited_node_ids"] == []

    # Lượt NGAY SAU lượt lỗi: history đưa vào contextualize() phải chứa
    # placeholder, KHÔNG chứa traceback/"[LỖI AGENT]" (xem plan5.md B1: tránh
    # bơm traceback vào prompt của các lượt sau).
    # ctx_calls[0] ứng với lượt 2 (lượt 1 miễn phí) -> lượt N ứng ctx_calls[N-2].
    # Lượt NGAY SAU lượt lỗi là FAIL_TURN + 1 -> ctx_calls[FAIL_TURN - 1].
    next_turn_ctx_messages = ctx_calls[FAIL_TURN - 1]
    human_content = next_turn_ctx_messages[1].content
    assert "[LỖI AGENT]" not in human_content
    assert "hệ thống không trả lời được câu này" in human_content


def test_all_30_turns_present_even_with_one_failure(
    tmp_path, monkeypatch, _patched_multiturn
):
    from slidegen_rag.evaluate import runs as runs_mod

    monkeypatch.setattr(runs_mod, "DATA_TEST_DIR", tmp_path)
    rp = RunPaths.for_new_run("chu_de_test")

    import asyncio

    log = asyncio.run(
        runner._run_multi_turn_session(
            "chu_de_test",
            _questions(),
            _references(),
            run_dir=rp,
            max_turns=5,
            lowcost=False,
            progress_callback=None,
        )
    )

    assert len(log["_results"]) == TOTAL_TURNS
    assert [r["num"] for r in log["_results"]] == list(range(1, TOTAL_TURNS + 1))
    assert rp.answer.exists()
    assert rp.chunks.exists()
