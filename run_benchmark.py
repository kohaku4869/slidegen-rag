"""One-off script: run pldc_v2 benchmark end-to-end (force_rerun=True) per
plan2.md verification step, and export scores/summary files."""
import sys

from slidegen_rag.evaluate.evaluate import (
    run_evaluation_for_topic,
    export_evaluation_results,
)


def main():
    topic = "pldc_v2"
    print(f"[run_benchmark] Bắt đầu chạy {topic} (force_rerun=True)...", flush=True)
    detail_df, summary, was_cached = run_evaluation_for_topic(topic, force_rerun=True)
    print(f"[run_benchmark] was_cached={was_cached}", flush=True)
    csv_path, json_path = export_evaluation_results(topic, detail_df, summary)
    print(f"[run_benchmark] Đã xuất: {csv_path}, {json_path}", flush=True)
    print("[run_benchmark] Summary:", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
