"""
Shim tương thích ngược cho dashboard đánh giá.

Trước đây toàn bộ dashboard nằm trong DUY NHẤT file này (2.249 dòng). Nay đã
tách thành package — xem `settings.py`, `io.py`, `metrics.py`, `judge.py`,
`runner.py`, `report.py`, `app.py`. File này chỉ còn re-export để lệnh chạy cũ
`python -m slidegen_rag.evaluate.evaluate` và các import cũ không vỡ.

Code MỚI nên import thẳng từ module con (ví dụ
`from slidegen_rag.evaluate.io import load_answers`) thay vì qua đây.
"""

from slidegen_rag.evaluate.app import (
    build_gradio_app,
    evaluate_all_topics_handler,
    evaluate_single_topic_handler,
    find_available_port,
    inspect_question_handler,
)
from slidegen_rag.evaluate.io import (
    get_available_topics,
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
    CRITERIA,
    JUDGE_PREAMBLE,
    AnswerEval,
    MetricEval,
    create_evaluator_llm,
    evaluate_single_answer_with_judge,
    evaluate_single_metric_with_judge,
    parse_metric_output,
)
from slidegen_rag.evaluate.metrics import (
    calculate_answer_coverage,
    calculate_keyword_coverage,
    calculate_keyword_coverage_strict,
)
from slidegen_rag.evaluate.report import (
    create_all_topics_chart,
    create_single_topic_chart,
    create_summary_table,
    export_evaluation_results,
    format_metric_html,
    generate_summary_kpi_html,
    get_color,
)
from slidegen_rag.evaluate.runner import (
    get_answers_and_chunks_for_topic,
    regenerate_answers_for_topic,
    run_evaluation_for_topic,
)
from slidegen_rag.evaluate.settings import (
    COVERAGE_AMBER,
    COVERAGE_GREEN,
    DATA_INDEX_DIR,
    DATA_TEST_DIR,
    GENERATE_STATE_FILENAME,
    ROOT_DIR,
    SCORE_AMBER,
    SCORE_GREEN,
)

__all__ = [
    "COVERAGE_AMBER",
    "COVERAGE_GREEN",
    "CRITERIA",
    "DATA_INDEX_DIR",
    "DATA_TEST_DIR",
    "GENERATE_STATE_FILENAME",
    "JUDGE_PREAMBLE",
    "ROOT_DIR",
    "SCORE_AMBER",
    "SCORE_GREEN",
    "AnswerEval",
    "MetricEval",
    "build_gradio_app",
    "calculate_answer_coverage",
    "calculate_keyword_coverage",
    "calculate_keyword_coverage_strict",
    "create_all_topics_chart",
    "create_evaluator_llm",
    "create_single_topic_chart",
    "create_summary_table",
    "evaluate_all_topics_handler",
    "evaluate_single_answer_with_judge",
    "evaluate_single_metric_with_judge",
    "evaluate_single_topic_handler",
    "export_evaluation_results",
    "find_available_port",
    "format_metric_html",
    "generate_summary_kpi_html",
    "get_answers_and_chunks_for_topic",
    "get_available_topics",
    "get_color",
    "inspect_question_handler",
    "load_answers",
    "load_chunks",
    "load_generate_state",
    "load_questions",
    "load_references",
    "load_trajectory_texts",
    "parse_metric_output",
    "regenerate_answers_for_topic",
    "run_evaluation_for_topic",
    "save_answers",
    "save_chunks",
    "save_generate_state",
]


if __name__ == "__main__":
    from slidegen_rag.evaluate.app import main

    main()
