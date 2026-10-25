"""Unit tests for TokenTracker (agent/models/llm.py): Gemini and local token
usage must be tracked SEPARATELY, and cost must be computed ONLY from the
Gemini portion — a local-model call must never inflate the estimated cost."""

from slidegen_rag.agent.models.llm import TokenTracker


def _usage(input_tokens: int, output_tokens: int, reasoning: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_token_details": {"reasoning": reasoning} if reasoning else {},
    }


def test_gemini_only_usage_tracked_and_priced():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)

    summary = tracker.get_summary("gemini-2.5-flash")

    assert summary["gemini_calls"] == 1
    assert summary["local_calls"] == 0
    assert summary["gemini_input_tokens"] == 1000
    assert summary["gemini_output_tokens"] == 500
    assert summary["gemini_total_tokens"] == 1500
    assert summary["total_cost_usd"] > 0


def test_local_only_usage_has_zero_cost():
    tracker = TokenTracker()
    tracker.add_usage(_usage(10_000, 5_000), is_local=True)

    summary = tracker.get_summary("gemini-2.5-flash")

    assert summary["gemini_calls"] == 0
    assert summary["local_calls"] == 1
    assert summary["local_input_tokens"] == 10_000
    assert summary["local_output_tokens"] == 5_000
    assert summary["local_total_tokens"] == 15_000
    # A local (self-hosted) call must contribute ZERO to the Gemini cost.
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_cost_vnd"] == 0.0


def test_mixed_usage_cost_reflects_only_gemini_portion():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    tracker.add_usage(_usage(50_000, 50_000), is_local=True)  # much bigger, but free

    gemini_only_tracker = TokenTracker()
    gemini_only_tracker.add_usage(_usage(1000, 500), is_local=False)

    mixed_summary = tracker.get_summary("gemini-2.5-flash")
    gemini_only_summary = gemini_only_tracker.get_summary("gemini-2.5-flash")

    # The huge local call must not change the estimated Gemini cost at all.
    assert mixed_summary["total_cost_usd"] == gemini_only_summary["total_cost_usd"]
    # But the backward-compat combined totals DO include both providers.
    assert mixed_summary["input_tokens"] == 51_000
    assert mixed_summary["output_tokens"] == 50_500
    assert mixed_summary["calls"] == 2


def test_reasoning_tokens_tracked_for_gemini_only():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500, reasoning=300), is_local=False)
    tracker.add_usage(_usage(200, 100, reasoning=999), is_local=True)

    summary = tracker.get_summary()
    assert summary["gemini_reasoning_tokens"] == 300


def test_reset_clears_both_gemini_and_local():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    tracker.add_usage(_usage(200, 100), is_local=True)

    tracker.reset()
    summary = tracker.get_summary()

    assert summary["calls"] == 0
    assert summary["gemini_total_tokens"] == 0
    assert summary["local_total_tokens"] == 0
    assert summary["total_cost_usd"] == 0.0


def test_format_report_mentions_local_only_when_present():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    report_no_local = tracker.format_report()
    assert "Local (miễn phí)" not in report_no_local

    tracker.add_usage(_usage(200, 100), is_local=True)
    report_with_local = tracker.format_report()
    assert "Local (miễn phí)" in report_with_local


def test_snapshot_returns_nine_raw_counters():
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500, reasoning=300), is_local=False)
    tracker.add_usage(_usage(200, 100), is_local=True)

    snap = tracker.snapshot()

    assert snap == {
        "gemini_calls": 1,
        "gemini_input_tokens": 1000,
        "gemini_output_tokens": 500,
        "gemini_reasoning_tokens": 300,
        "gemini_total_tokens": 1500,
        "local_calls": 1,
        "local_input_tokens": 200,
        "local_output_tokens": 100,
        "local_total_tokens": 300,
    }


def test_summary_from_delta_matches_get_summary_shape():
    tracker = TokenTracker()
    before = tracker.snapshot()
    tracker.add_usage(_usage(1000, 500), is_local=False)
    tracker.add_usage(_usage(200, 100), is_local=True)
    after = tracker.snapshot()

    delta_summary = TokenTracker.summary_from_delta(before, after, "gemini-2.5-flash")
    direct_summary = tracker.get_summary("gemini-2.5-flash")

    assert delta_summary == direct_summary


def test_summary_from_delta_isolates_a_span_without_resetting():
    """The whole point of delta accounting: code that runs BEFORE the span
    (e.g. contextualize()) must not be wiped, and code that runs AFTER
    capturing `after` must not leak into this span's numbers."""
    tracker = TokenTracker()
    tracker.add_usage(_usage(1000, 500), is_local=False)  # e.g. contextualize()

    before = tracker.snapshot()
    tracker.add_usage(_usage(300, 150), is_local=False)  # the span being measured
    after = tracker.snapshot()

    tracker.add_usage(_usage(9999, 9999), is_local=False)  # e.g. next turn's call

    span_summary = TokenTracker.summary_from_delta(before, after)
    assert span_summary["gemini_calls"] == 1
    assert span_summary["gemini_input_tokens"] == 300
    assert span_summary["gemini_output_tokens"] == 150

    # Nothing was reset: the tracker's cumulative totals still include
    # everything, including calls before `before` and after `after`.
    full_summary = tracker.get_summary()
    assert full_summary["gemini_calls"] == 3
    assert full_summary["gemini_input_tokens"] == 1000 + 300 + 9999
