"""Judge token accounting: evaluate_single_metric_with_judge must record usage
into judge.judge_tracker — a tracker SEPARATE from LLMModel.tracker (the judge
builds its own ChatOpenAI/ChatGoogleGenerativeAI, never touching LLMModel) — so
judge cost becomes visible without ever leaking into the agent's own numbers.
"""

import pytest

from slidegen_rag.evaluate import judge as judge_module


class _FakeAIMessage:
    def __init__(self, content: str, usage: dict | None):
        self.content = content
        self.usage_metadata = usage


class _FakeLLM:
    """Mimics langchain's `.ainvoke` returning an AIMessage with usage_metadata."""

    def __init__(self, content: str, usage: dict | None, *, fail: bool = False):
        self._content = content
        self._usage = usage
        self._fail = fail
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return _FakeAIMessage(self._content, self._usage)


VALID_METRIC_JSON = '{"score": 8.5, "feedback": "ổn"}'


def _usage(inp: int, out: int) -> dict:
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


@pytest.fixture(autouse=True)
def _reset_judge_tracker():
    judge_module.judge_tracker.reset()
    yield
    judge_module.judge_tracker.reset()


@pytest.mark.asyncio
async def test_primary_success_records_as_local():
    llm = _FakeLLM(VALID_METRIC_JSON, _usage(100, 50))
    fallback = _FakeLLM(VALID_METRIC_JSON, _usage(999, 999))

    result = await judge_module.evaluate_single_metric_with_judge(
        llm, fallback, judge_module.CRITERIA[0], "q", "ref", "ans", []
    )

    assert result is not None
    assert fallback.calls == 0  # primary succeeded, fallback never called
    summary = judge_module.judge_tracker.get_summary()
    assert summary["local_calls"] == 1
    assert summary["local_input_tokens"] == 100
    assert summary["local_output_tokens"] == 50
    assert summary["gemini_calls"] == 0


@pytest.mark.asyncio
async def test_fallback_used_records_as_gemini():
    llm = _FakeLLM("not-json garbage", _usage(10, 5))  # parse fails -> fallback
    fallback = _FakeLLM(VALID_METRIC_JSON, _usage(200, 100))

    result = await judge_module.evaluate_single_metric_with_judge(
        llm, fallback, judge_module.CRITERIA[0], "q", "ref", "ans", []
    )

    assert result is not None
    summary = judge_module.judge_tracker.get_summary()
    # Primary's failed-parse attempt still consumed tokens -> counted local.
    assert summary["local_calls"] == 1
    assert summary["local_input_tokens"] == 10
    # Fallback (Gemini) succeeded.
    assert summary["gemini_calls"] == 1
    assert summary["gemini_input_tokens"] == 200
    assert summary["gemini_output_tokens"] == 100


@pytest.mark.asyncio
async def test_judge_tracker_never_touches_agent_tracker():
    from slidegen_rag.agent.models.llm import LLMModel

    agent_before = LLMModel.tracker.snapshot()

    llm = _FakeLLM(VALID_METRIC_JSON, _usage(100, 50))
    fallback = _FakeLLM(VALID_METRIC_JSON, _usage(1, 1))
    await judge_module.evaluate_single_metric_with_judge(
        llm, fallback, judge_module.CRITERIA[0], "q", "ref", "ans", []
    )

    agent_after = LLMModel.tracker.snapshot()
    assert agent_before == agent_after
