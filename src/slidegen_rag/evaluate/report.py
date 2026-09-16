"""
Kết xuất kết quả đánh giá: thẻ KPI HTML, biểu đồ matplotlib, bảng tổng hợp và
ghi file CSV/JSON làm baseline đối chiếu giữa các lần chạy.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slidegen_rag.evaluate.settings import (
    COVERAGE_AMBER,
    COVERAGE_GREEN,
    SCORE_AMBER,
    SCORE_GREEN,
)

logger = logging.getLogger(__name__)


def get_color(value: Optional[float], is_percentage: bool = False) -> str:
    """Xác định màu sắc theo ngưỡng điểm (Thang 10 hoặc %)."""
    if value is None:
        return "#94a3b8"
    if is_percentage:
        if value >= COVERAGE_GREEN:
            return "#10b981"
        elif value >= COVERAGE_AMBER:
            return "#f59e0b"
        else:
            return "#ef4444"
    else:
        if value >= SCORE_GREEN:
            return "#10b981"
        elif value >= SCORE_AMBER:
            return "#f59e0b"
        else:
            return "#ef4444"


def format_metric_html(
    label: str,
    value: Optional[float],
    is_percentage: bool = False,
    is_count: bool = False,
) -> str:
    """Tạo khối thẻ HTML KPI card với viền màu động sang trọng."""
    if value is None:
        color = "#94a3b8"
        val_str = "N/A"
    elif is_count:
        color = "#3b82f6"
        val_str = f"{value:.1f} chunks"
    elif is_percentage:
        color = get_color(value, is_percentage=True)
        val_str = f"{value:.1f}%"
    else:
        color = get_color(value, is_percentage=False)
        val_str = f"{value:.2f}/10"

    return f"""
    <div style="margin: 6px 0; padding: 12px 16px; background-color: #f8fafc; border-radius: 8px; border-left: 6px solid {color}; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>
        <div style="font-size: 24px; font-weight: 800; color: {color}; line-height: 1.1;">{val_str}</div>
    </div>
    """


def generate_summary_kpi_html(
    summary: dict, topic_name: str = "", was_cached: bool = False
) -> str:
    """Tạo toàn bộ khối KPI metrics HTML cho Dashboard."""
    cache_badge = (
        '<span style="background-color: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">⚡ Cache (answer.txt & chunks.txt)</span>'
        if was_cached
        else '<span style="background-color: #fef3c7; color: #b45309; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">🤖 Live Agent Run</span>'
    )

    failed_cnt = summary.get("failed_count", 0)
    failed_badge = (
        f' | <span style="color: #ef4444; font-weight: 700;">⚠️ {failed_cnt} câu lỗi (không tính vào điểm)</span>'
        if failed_cnt > 0
        else ""
    )

    return f"""
    <div style="padding: 10px 0;">
        <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 15px; font-weight: 700; color: #1e293b;">📊 Tổng quan đánh giá: {topic_name}</span>
            {cache_badge}
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
            {format_metric_html("🏆 Tổng điểm (Total Score)", summary.get("total_score"))}
            {format_metric_html("🎯 Độ chính xác (Accuracy)", summary.get("accuracy"))}
            {format_metric_html("📚 Độ đầy đủ (Completeness)", summary.get("completeness"))}
            {format_metric_html("🎯 Độ liên quan (Relevance)", summary.get("relevance"))}
            {format_metric_html("🛡️ Trung thực (Faithfulness)", summary.get("faithfulness"))}
            {format_metric_html("🔍 Phủ từ khóa (Keyword Coverage - legacy)", summary.get("coverage_pct"), is_percentage=True)}
            {format_metric_html("🔍 Phủ từ khóa (Keyword Coverage - strict)", summary.get("coverage_pct_strict"), is_percentage=True)}
        </div>
        <div style="margin-top: 10px; padding: 8px 12px; background-color: #f1f5f9; border-radius: 6px; font-size: 12px; color: #475569; text-align: center;">
            Tổng số câu hỏi: <b>{summary.get("total_questions", 0)}</b> (Thành công: <b>{summary.get("success_count", 0)}</b>{failed_badge}) | Trung bình Context: <b>{summary.get("avg_chunks", 0.0):.1f} chunks/câu</b>
        </div>
    </div>
    """


def create_single_topic_chart(summary: dict, topic: str) -> plt.Figure:
    """Tạo biểu đồ cột thể hiện các chỉ số của một chủ đề."""
    metrics = ["Accuracy", "Completeness", "Relevance", "Faithfulness", "Total Score"]
    scores = [
        summary.get("accuracy", 0.0),
        summary.get("completeness", 0.0),
        summary.get("relevance", 0.0),
        summary.get("faithfulness", 0.0),
        summary.get("total_score", 0.0),
    ]

    colors = [get_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    bars = ax.bar(
        metrics, scores, color=colors, width=0.55, edgecolor="white", linewidth=1.5
    )

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            color="#1e293b",
        )

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm (Thang 10)", fontsize=11, fontweight="600")
    ax.set_title(
        f"Phân tích chất lượng RAG — Chủ đề: {topic}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.axhline(
        y=SCORE_GREEN,
        color="#10b981",
        linestyle="--",
        alpha=0.6,
        label="Mục tiêu (>= 8.5)",
    )
    ax.axhline(
        y=SCORE_AMBER,
        color="#f59e0b",
        linestyle=":",
        alpha=0.6,
        label="Ngưỡng đạt (>= 7.0)",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_all_topics_chart(all_summaries: dict[str, dict]) -> plt.Figure:
    """Tạo biểu đồ so sánh các chủ đề."""
    topics = list(all_summaries.keys())
    if not topics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        return fig

    x = np.arange(len(topics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(topics) * 2.5), 5), dpi=100)

    acc = [all_summaries[t].get("accuracy", 0.0) for t in topics]
    comp = [all_summaries[t].get("completeness", 0.0) for t in topics]
    rel = [all_summaries[t].get("relevance", 0.0) for t in topics]
    faith = [all_summaries[t].get("faithfulness", 0.0) for t in topics]

    ax.bar(
        x - 1.5 * width,
        acc,
        width,
        label="Accuracy",
        color="#3b82f6",
        edgecolor="white",
    )
    ax.bar(
        x - 0.5 * width,
        comp,
        width,
        label="Completeness",
        color="#8b5cf6",
        edgecolor="white",
    )
    ax.bar(
        x + 0.5 * width,
        rel,
        width,
        label="Relevance",
        color="#06b6d4",
        edgecolor="white",
    )
    ax.bar(
        x + 1.5 * width,
        faith,
        width,
        label="Faithfulness",
        color="#10b981",
        edgecolor="white",
    )

    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Điểm (Thang 10)", fontsize=11, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, fontsize=11, fontweight="600")
    ax.set_title(
        "So sánh các chỉ số RAG giữa các chủ đề", fontsize=14, fontweight="bold", pad=12
    )
    ax.axhline(
        y=SCORE_GREEN,
        color="#10b981",
        linestyle="--",
        alpha=0.5,
        label="Mục tiêu (8.5)",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def create_summary_table(all_summaries: dict[str, dict]) -> pd.DataFrame:
    """Tạo bảng tổng hợp điểm cho các chủ đề."""
    rows = []
    for topic, summary in all_summaries.items():
        rows.append(
            {
                "Chủ đề (Topic)": topic,
                "Accuracy": round(summary.get("accuracy", 0.0), 2),
                "Completeness": round(summary.get("completeness", 0.0), 2),
                "Relevance": round(summary.get("relevance", 0.0), 2),
                "Faithfulness": round(summary.get("faithfulness", 0.0), 2),
                "Total Score": round(summary.get("total_score", 0.0), 2),
                "Keyword Coverage legacy (%)": f"{summary.get('coverage_pct', 0.0):.1f}%",
                "Keyword Coverage strict (%)": f"{summary.get('coverage_pct_strict', 0.0):.1f}%",
                "Avg Chunks": round(summary.get("avg_chunks", 0.0), 1),
                "Số câu thành công": f"{summary.get('success_count', 0)}/{summary.get('total_questions', 0)}",
            }
        )
    return pd.DataFrame(rows)


def export_evaluation_results(run_paths, detail_df: pd.DataFrame, summary: dict) -> tuple[Path, Path]:
    """
    Ghi kết quả đánh giá vào `run_paths.scores`/`run_paths.summary` — tức
    `data/test/<topic>/runs/<run_id>/scores.csv`/`summary.json` (xem
    plan5.md Phần C). `run_id` được sinh lúc BẮT ĐẦU run (`RunPaths.for_new_run`
    trong runner.py), không phải ở đây — artifact và điểm số luôn cùng một
    id, không còn cần đối chiếu bằng tay như layout `scores_<ts>.csv` cũ.

    Args:
        run_paths: `evaluate.runs.RunPaths` của run đang ghi.

    Returns:
        (csv_path, json_path)
    """
    run_paths.root.mkdir(parents=True, exist_ok=True)

    export_cols = [
        c for c in detail_df.columns if c not in ("used_chunks", "traj_text")
    ]
    detail_df[export_cols].to_csv(run_paths.scores, index=False, encoding="utf-8-sig")

    with open(run_paths.summary, "w", encoding="utf-8") as f:
        json.dump(
            {"topic": run_paths.topic, "run_id": run_paths.run_id, **summary},
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        f"[{run_paths.topic}] Đã xuất kết quả run {run_paths.run_id}: "
        f"{run_paths.scores.name}, {run_paths.summary.name}"
    )
    return run_paths.scores, run_paths.summary
