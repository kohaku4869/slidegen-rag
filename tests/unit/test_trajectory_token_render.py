"""Unit tests for render_trajectory's token summary line: must show
Gemini-only token/cost (not mixed with local), and must fall back gracefully
to the old combined `token_usage` shape for trajectories recorded before the
Gemini/local split (see agent/models/llm.py::TokenTracker)."""

from datetime import UTC, datetime

from slidegen_rag.agent.observability.trajectory import Trajectory, render_trajectory


def _traj(token_usage: dict) -> Trajectory:
    return Trajectory(
        query="câu hỏi test",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        token_usage=token_usage,
    )


def test_renders_gemini_specific_line_with_new_shape():
    text = render_trajectory(
        _traj(
            {
                "gemini_input_tokens": 1000,
                "gemini_output_tokens": 500,
                "gemini_total_tokens": 1500,
                "local_total_tokens": 0,
                "local_calls": 0,
                "total_cost_usd": 0.001,
                "total_cost_vnd": 25.4,
            }
        )
    )
    assert "Tokens Gemini: in=1,000 out=500 tổng=1,500" in text
    assert "Tokens Local" not in text


def test_renders_local_line_when_local_calls_present():
    text = render_trajectory(
        _traj(
            {
                "gemini_input_tokens": 1000,
                "gemini_output_tokens": 500,
                "gemini_total_tokens": 1500,
                "local_total_tokens": 800,
                "local_calls": 4,
                "total_cost_usd": 0.001,
                "total_cost_vnd": 25.4,
            }
        )
    )
    assert "Tokens Local (miễn phí): tổng=800 (4 lượt)" in text


def test_falls_back_to_legacy_combined_shape():
    # Old token_usage (pre-split) only has the combined keys.
    text = render_trajectory(
        _traj(
            {
                "input_tokens": 2000,
                "output_tokens": 300,
                "total_tokens": 2300,
                "total_cost_usd": 0.002,
                "total_cost_vnd": 50.8,
            }
        )
    )
    assert "Tokens Gemini: in=2,000 out=300 tổng=2,300" in text


def test_empty_token_usage_omits_tokens_line():
    text = render_trajectory(_traj({}))
    assert "Tokens" not in text
