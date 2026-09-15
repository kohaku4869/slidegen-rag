"""
Giao diện Gradio của dashboard đánh giá và các handler nối nút bấm với
runner/report. Đây là điểm vào: `python -m slidegen_rag.evaluate.app`.
"""

import logging
import re
import socket
import traceback
from typing import Optional

import gradio as gr
import pandas as pd

from slidegen_rag.evaluate.io import get_available_topics
from slidegen_rag.evaluate.report import (
    create_all_topics_chart,
    create_single_topic_chart,
    create_summary_table,
    export_evaluation_results,
    generate_summary_kpi_html,
)
from slidegen_rag.evaluate.runner import run_evaluation_for_topic

logger = logging.getLogger(__name__)

_current_detail_df: Optional[pd.DataFrame] = None


def evaluate_single_topic_handler(
    topic: str,
    force_rerun: bool,
    lowcost: bool = False,
    regenerate_only: bool = False,
    progress=gr.Progress(),
):
    """Handler đánh giá một chủ đề."""
    global _current_detail_df

    if not topic:
        return (
            None,
            None,
            "<p>⚠️ Vui lòng chọn một chủ đề!</p>",
            gr.Dropdown(choices=[]),
            "⚠️ Vui lòng chọn một chủ đề!",
        )

    try:
        progress(0.1, desc=f"Đang chuẩn bị dữ liệu cho chủ đề: {topic}...")

        def cb(msg):
            progress(0.4, desc=msg)

        detail_df, summary, was_cached = run_evaluation_for_topic(
            topic,
            force_rerun=force_rerun,
            progress_callback=cb,
            lowcost=lowcost,
            regenerate_only=regenerate_only,
        )

        _current_detail_df = detail_df

        progress(0.85, desc="Đang tạo biểu đồ & báo cáo...")

        kpi_html = generate_summary_kpi_html(
            summary, topic_name=topic, was_cached=was_cached
        )
        chart = create_single_topic_chart(summary, topic)

        display_cols = [
            "STT",
            "question",
            "response",
            "accuracy",
            "completeness",
            "relevance",
            "faithfulness",
            "total_score",
            "coverage_pct",
            "coverage_pct_strict",
            "chunks_count",
            "feedback",
        ]
        display_df = detail_df[display_cols].copy()
        display_df = display_df.fillna("-")

        export_evaluation_results(topic, detail_df, summary)

        q_choices = [
            f"Câu {row['STT']}: {str(row['question'])[:60]}..."
            for _, row in detail_df.iterrows()
        ]
        first_choice = q_choices[0] if q_choices else None

        succ = summary.get("success_count", 0)
        tot = summary.get("total_questions", 0)
        fail = summary.get("failed_count", 0)
        fail_msg = f" (⚠️ {fail} câu lỗi không tính vào điểm tổng)" if fail > 0 else ""
        mode_msg = (
            " — ♻️ chế độ CHỈ GENERATE: chunk giữ nguyên từ lần chạy đầy đủ trước, "
            "nên coverage/avg_chunks không phải số đo mới"
            if regenerate_only
            else ""
        )
        status_msg = (
            f"✅ Hoàn tất đánh giá chủ đề **{topic}** "
            f"(Thành công {succ}/{tot} câu{fail_msg}){mode_msg}!"
        )

        progress(1.0, desc="Hoàn tất!")

        return (
            chart,
            display_df,
            kpi_html,
            gr.Dropdown(choices=q_choices, value=first_choice),
            status_msg,
        )

    except Exception as e:
        logger.error(f"Lỗi đánh giá {topic}: {e}", exc_info=True)
        err_msg = f"❌ Lỗi: {e}\n\n{traceback.format_exc()}"
        return (
            None,
            None,
            f"<div style='color: red;'>{err_msg}</div>",
            gr.Dropdown(choices=[]),
            err_msg,
        )


def inspect_question_handler(selected_q_text: str):
    """Handler hiển thị chi tiết câu hỏi, đáp án, chunks, feedback và trajectory."""
    global _current_detail_df
    if _current_detail_df is None or not selected_q_text:
        return "", "", "", ""

    match = re.match(r"^Câu\s*(\d+):", selected_q_text)
    if not match:
        return "", "", "", ""

    stt = int(match.group(1))
    row = _current_detail_df[_current_detail_df["STT"] == stt]
    if row.empty:
        return "", "", "", ""

    item = row.iloc[0]

    col1_content = f"""### ❓ Câu hỏi #{item["STT"]}
**{item["question"]}**

---
### 📖 Đáp án mẫu (Reference Answer)
{item["reference"] if item["reference"] else "*(Không có đáp án mẫu)*"}

---
### 🤖 Câu trả lời của CRAG Agent
{item["response"]}
"""

    chunks_list = item.get("used_chunks", [])
    if not chunks_list:
        chunks_content = "*(Không có context chunk nào được sử dụng)*"
    else:
        chunks_blocks = []
        for c_idx, chunk_text in enumerate(chunks_list, 1):
            chunks_blocks.append(f"**[Chunk {c_idx}]**\n```markdown\n{chunk_text}\n```")
        chunks_content = "\n\n".join(chunks_blocks)

    col2_content = f"""### 📄 Các Chunks tài liệu cuối cùng đã dùng ({len(chunks_list)} chunks)
{chunks_content}
"""

    acc_str = f"{item['accuracy']}/10" if pd.notna(item["accuracy"]) else "N/A"
    comp_str = f"{item['completeness']}/10" if pd.notna(item["completeness"]) else "N/A"
    rel_str = f"{item['relevance']}/10" if pd.notna(item["relevance"]) else "N/A"
    faith_str = (
        f"{item['faithfulness']}/10" if pd.notna(item["faithfulness"]) else "N/A"
    )
    tot_str = f"{item['total_score']}/10" if pd.notna(item["total_score"]) else "N/A"

    col3_content = f"""### ⚖️ Đánh giá từ LLM Judge (Thang 10)
- **Độ chính xác (Accuracy)**: `{acc_str}`
- **Độ đầy đủ (Completeness)**: `{comp_str}`
- **Độ liên quan (Relevance)**: `{rel_str}`
- **Trung thực / Chống ảo giác (Faithfulness)**: `{faith_str}`
- **🏆 Tổng điểm (Total Score)**: `{tot_str}`
- **Độ phủ từ khóa - legacy (Keyword Coverage)**: `{item["coverage_pct"]}%`
- **Độ phủ từ khóa - strict (Keyword Coverage)**: `{item.get("coverage_pct_strict", "N/A")}%`

---
### 📝 Nhận xét chi tiết (Feedback)
{item["feedback"]}
"""

    traj_text = item.get("traj_text") or "*(Không có trajectory)*"
    col4_content = f"""### 🧭 Trajectory (quỹ đạo thực thi Agent)
```
{traj_text}
```
"""

    return col1_content, col2_content, col3_content, col4_content


def evaluate_all_topics_handler(
    force_rerun: bool, lowcost: bool = False, progress=gr.Progress()
):
    """Handler đánh giá toàn bộ các chủ đề."""
    topics = get_available_topics()
    if not topics:
        return (
            None,
            None,
            None,
            "<p>⚠️ Không tìm thấy chủ đề nào trong data/test/</p>",
            "⚠️ Không có dữ liệu.",
        )

    all_summaries = {}
    all_details = []
    total_questions = 0
    total_success = 0
    total_failed = 0

    for t_idx, topic in enumerate(topics):
        try:
            base_prog = t_idx / len(topics)
            progress(
                base_prog, desc=f"[{t_idx + 1}/{len(topics)}] Đang xử lý: {topic}..."
            )

            def cb(msg):
                progress(base_prog + 0.5 / len(topics), desc=msg)

            detail_df, summary, was_cached = run_evaluation_for_topic(
                topic, force_rerun=force_rerun, progress_callback=cb, lowcost=lowcost
            )

            all_summaries[topic] = summary
            all_details.append(detail_df)
            total_questions += summary.get("total_questions", 0)
            total_success += summary.get("success_count", 0)
            total_failed += summary.get("failed_count", 0)

            export_evaluation_results(topic, detail_df, summary)

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá chủ đề {topic}: {e}", exc_info=True)
            all_summaries[topic] = {
                "accuracy": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "faithfulness": 0.0,
                "total_score": 0.0,
                "coverage_pct": 0.0,
                "coverage_pct_strict": 0.0,
                "avg_chunks": 0.0,
                "total_questions": 0,
                "success_count": 0,
                "failed_count": 0,
                "error": str(e),
            }

    progress(0.9, desc="Đang tạo biểu đồ tổng hợp...")

    chart = create_all_topics_chart(all_summaries)
    summary_df = create_summary_table(all_summaries)

    if all_details:
        combined_df = pd.concat(all_details, ignore_index=True)
        display_cols = [
            "STT",
            "topic",
            "question",
            "response",
            "accuracy",
            "completeness",
            "relevance",
            "faithfulness",
            "total_score",
            "coverage_pct",
            "coverage_pct_strict",
            "chunks_count",
            "feedback",
        ]
        detail_table_df = combined_df[
            [c for c in display_cols if c in combined_df.columns]
        ].copy()
        detail_table_df = detail_table_df.fillna("-")
    else:
        detail_table_df = pd.DataFrame()

    valid_summaries = [
        s
        for s in all_summaries.values()
        if "error" not in s and s.get("success_count", 0) > 0
    ]
    if valid_summaries:
        overall_summary = {
            "total_score": sum(s["total_score"] for s in valid_summaries)
            / len(valid_summaries),
            "accuracy": sum(s["accuracy"] for s in valid_summaries)
            / len(valid_summaries),
            "completeness": sum(s["completeness"] for s in valid_summaries)
            / len(valid_summaries),
            "relevance": sum(s["relevance"] for s in valid_summaries)
            / len(valid_summaries),
            "faithfulness": sum(s["faithfulness"] for s in valid_summaries)
            / len(valid_summaries),
            "coverage_pct": sum(s["coverage_pct"] for s in valid_summaries)
            / len(valid_summaries),
            "coverage_pct_strict": sum(
                s.get("coverage_pct_strict", 0.0) for s in valid_summaries
            )
            / len(valid_summaries),
            "avg_chunks": sum(s["avg_chunks"] for s in valid_summaries)
            / len(valid_summaries),
            "total_questions": total_questions,
            "success_count": total_success,
            "failed_count": total_failed,
        }
    else:
        overall_summary = {}

    kpi_html = generate_summary_kpi_html(
        overall_summary,
        topic_name="Toàn hệ thống (All Topics)",
        was_cached=not force_rerun,
    )

    fail_msg = (
        f" (⚠️ {total_failed} câu lỗi không tính vào điểm tổng)"
        if total_failed > 0
        else ""
    )
    status_msg = f"✅ Đã hoàn tất đánh giá {len(valid_summaries)}/{len(topics)} chủ đề (Thành công {total_success}/{total_questions} câu{fail_msg})!"

    progress(1.0, desc="Hoàn tất!")

    return chart, summary_df, detail_table_df, kpi_html, status_msg


# ── Giao diện Gradio (theme & blocks) ─────────────────────────────────────


def build_gradio_app():
    """Khởi tạo giao diện Dashboard Gradio."""
    topics = get_available_topics()
    topic_choices = topics if topics else ["(Không có chủ đề)"]

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        font=["Inter", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="SLIDEGEN-RAG Evaluation Dashboard") as app:
        gr.Markdown(
            """
            # 📊 SLIDEGEN-RAG — Dashboard Đánh Giá Chất Lượng
            Hệ thống đánh giá toàn diện **CRAG Agent** sử dụng phương pháp **LLM-as-a-Judge** trực tiếp (Thang điểm 10).
            
            **Các tiêu chí đánh giá cốt lõi:**
            - 🎯 **Accuracy (Độ chính xác)**: So sánh với đáp án mẫu, kiểm tra tính đúng đắn của thông tin.
            - 📚 **Completeness (Độ đầy đủ)**: Bao quát toàn bộ các ý cốt lõi và căn cứ trong tài liệu.
            - 🎯 **Relevance (Độ liên quan)**: Bám sát câu hỏi, không lan man, không chứa thông tin thừa.
            - 🛡️ **Faithfulness (Chống ảo giác / Hallucination)**: Đảm bảo câu trả lời chỉ xuất phát từ các context chunks đã sử dụng.
            """
        )

        with gr.Tabs():
            # ── TAB 1: Đánh giá theo chủ đề ───────────────────────────────
            with gr.TabItem("🎯 Đánh giá theo chủ đề", id="tab_single"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=2):
                        topic_dropdown = gr.Dropdown(
                            choices=topic_choices,
                            value=topic_choices[0] if topic_choices else None,
                            label="Chọn chủ đề kiểm thử (Topic)",
                            interactive=True,
                        )
                    with gr.Column(scale=2):
                        chk_force_single = gr.Checkbox(
                            label="🔄 Buộc chạy lại Agent (Bỏ qua cache answer/chunks)",
                            value=False,
                        )
                    with gr.Column(scale=2):
                        chk_lowcost_single = gr.Checkbox(
                            label="💰 Lowcost (generate_node dùng model tự host thay Gemini)",
                            value=False,
                        )
                    with gr.Column(scale=1):
                        btn_eval_single = gr.Button(
                            "🚀 Bắt đầu đánh giá",
                            variant="primary",
                            size="lg",
                        )
                        btn_regen_single = gr.Button(
                            "♻️ Chỉ chạy lại Generate",
                            variant="secondary",
                            size="lg",
                        )

                gr.Markdown(
                    "*♻️ **Chỉ chạy lại Generate**: bỏ qua retrieve/grade, sinh lại câu "
                    "trả lời trên đúng bộ chunk trong `chunks.txt` rồi chấm — dùng khi "
                    "sửa prompt/model của riêng bước tổng hợp. Nhanh và rẻ hơn hẳn chạy "
                    "đầy đủ, và do chunk không đổi nên chênh lệch điểm là do bước "
                    "generate chứ không lẫn nhiễu retrieval. Ghi đè `answer.txt`, giữ "
                    "nguyên `chunks.txt`.*"
                )

                status_single = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_single = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Bắt đầu đánh giá' để hiển thị các chỉ số</div>"
                        )
                    with gr.Column(scale=1):
                        chart_single = gr.Plot(label="Biểu đồ phân tích chất lượng")

                gr.Markdown("### 📋 Bảng điểm chi tiết từng câu hỏi")
                table_single = gr.Dataframe(
                    label="Chi tiết kết quả đánh giá",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔍 Soi chi tiết từng câu hỏi (Question Inspector)")
                with gr.Row():
                    inspect_dropdown = gr.Dropdown(
                        choices=[],
                        label="Chọn câu hỏi cần soi chi tiết",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_col1 = gr.Markdown(
                            "*(Chọn một câu hỏi ở trên để xem chi tiết)*"
                        )
                    with gr.Column(scale=1):
                        inspect_col2 = gr.Markdown("")
                    with gr.Column(scale=1):
                        inspect_col3 = gr.Markdown("")
                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_col4 = gr.Markdown(
                            label="Trajectory (quỹ đạo thực thi Agent)",
                        )

                # Hằng số truyền cho tham số regenerate_only của handler — 2 nút
                # dùng chung 1 handler, chỉ khác giá trị cờ này.
                state_full_run = gr.State(False)
                state_regen_only = gr.State(True)

                single_outputs = [
                    chart_single,
                    table_single,
                    kpi_single,
                    inspect_dropdown,
                    status_single,
                ]

                btn_eval_single.click(
                    fn=evaluate_single_topic_handler,
                    inputs=[
                        topic_dropdown,
                        chk_force_single,
                        chk_lowcost_single,
                        state_full_run,
                    ],
                    outputs=single_outputs,
                )

                btn_regen_single.click(
                    fn=evaluate_single_topic_handler,
                    inputs=[
                        topic_dropdown,
                        chk_force_single,
                        chk_lowcost_single,
                        state_regen_only,
                    ],
                    outputs=single_outputs,
                )

                inspect_dropdown.change(
                    fn=inspect_question_handler,
                    inputs=[inspect_dropdown],
                    outputs=[inspect_col1, inspect_col2, inspect_col3, inspect_col4],
                )

            # ── TAB 2: Đánh giá tất cả chủ đề ─────────────────────────────
            with gr.TabItem("📋 Đánh giá tất cả chủ đề", id="tab_all"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=3):
                        gr.Markdown(
                            f"Sẽ đánh giá tất cả **{len(topics)}** chủ đề trong `data/test/`: **{', '.join(topics) if topics else 'N/A'}**"
                        )
                    with gr.Column(scale=2):
                        chk_force_all = gr.Checkbox(
                            label="🔄 Buộc chạy lại Agent cho tất cả (Bỏ qua cache)",
                            value=False,
                        )
                    with gr.Column(scale=2):
                        chk_lowcost_all = gr.Checkbox(
                            label="💰 Lowcost (generate_node dùng model tự host thay Gemini)",
                            value=False,
                        )
                    with gr.Column(scale=1):
                        btn_eval_all = gr.Button(
                            "🚀 Đánh giá tất cả",
                            variant="primary",
                            size="lg",
                        )

                status_all = gr.Markdown("", label="Trạng thái")

                with gr.Row():
                    with gr.Column(scale=1):
                        kpi_all = gr.HTML(
                            "<div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>Nhấn 'Đánh giá tất cả' để hiển thị tổng hợp</div>"
                        )
                    with gr.Column(scale=1):
                        chart_all = gr.Plot(label="Biểu đồ so sánh giữa các chủ đề")

                gr.Markdown("### 📊 Bảng tổng hợp điểm các chủ đề")
                summary_table_all = gr.Dataframe(
                    label="Tổng hợp điểm",
                    interactive=False,
                )

                gr.Markdown("### 📝 Bảng chi tiết toàn bộ câu hỏi")
                detail_table_all = gr.Dataframe(
                    label="Chi tiết toàn bộ câu hỏi",
                    interactive=False,
                    wrap=True,
                )

                btn_eval_all.click(
                    fn=evaluate_all_topics_handler,
                    inputs=[chk_force_all, chk_lowcost_all],
                    outputs=[
                        chart_all,
                        summary_table_all,
                        detail_table_all,
                        kpi_all,
                        status_all,
                    ],
                )

    return app


def find_available_port(start_port: int = 7860, max_search: int = 100) -> int:
    """Tìm port khả dụng để chạy Gradio app mà không bị lỗi đụng độ port."""
    for port in range(start_port, start_port + max_search):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


# ── MAIN ENTRYPOINT ───────────────────────────────────────────────────────


def main():
    """Dựng app rồi mở dashboard. Tách thành hàm để `evaluate.py` (shim tương
    thích ngược) gọi lại được, thay vì nhân đôi khối khởi chạy."""
    app = build_gradio_app()
    port = find_available_port(7860)
    logger.info(f"Khởi chạy Evaluation Dashboard tại: http://localhost:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
