"""
main.py — Giao diện Hỏi Đáp (Q&A) tương tác sử dụng Corrective RAG Agent và cây phân cấp
(hierarchical tree) dựng từ cấu trúc heading của tài liệu Markdown nguồn.

Cách sử dụng:
    # 1. Chế độ chat tương tác với file index mặc định (hoặc chọn từ danh sách)
    uv run slidegen-ask

    # 2. Chỉ định file index cụ thể
    uv run slidegen-ask --index data/index/pldc.pkl

    # 3. Hỏi trực tiếp một câu không cần vào vòng lặp chat
    uv run slidegen-ask --query "Quy phạm pháp luật là gì?"

    # 4. Bật chế độ chi tiết (xem các bước Agent suy luận và gọi tools)
    uv run slidegen-ask --verbose
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ── Fix encoding console trên Windows ─────────────────────────────────────────
if sys.platform == "win32":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Định dạng logging ────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_ra(index_path: Path):
    """Khởi tạo RetrievalAugmentation và load cây phân cấp từ file .pkl."""
    from slidegen_rag.config import TreeConfig, RetrieverConfig
    from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
    from slidegen_rag.tree.retriever.tree_retriever import TreeRetrieverConfig
    cfg_tree = TreeConfig

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file index tại: {index_path}")

    logger.info(f"Đang tải cây từ: {index_path}")
    embedding_model = EmbeddingModel(model_name=cfg_tree.EMBEDDING_MODEL_NAME)

    tb_config = TreeBuilderConfig(
        summarization_length=cfg_tree.SUMMARIZATION_LENGTH,
        embedding_models={cfg_tree.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=cfg_tree.CLUSTER_EMBEDDING_MODEL,
    )

    tr_config = TreeRetrieverConfig(
        threshold=RetrieverConfig.THRESHOLD,
        top_k=RetrieverConfig.TOP_K,
        selection_mode=RetrieverConfig.SELECTION_MODE,
        context_embedding_model=RetrieverConfig.CONTEXT_EMBEDDING_MODEL,
        num_layers=RetrieverConfig.NUM_LAYERS,
        start_layer=RetrieverConfig.START_LAYER,
    )

    ra_config = RetrievalAugmentationConfig(
        tree_builder_config=tb_config,
        tree_retriever_config=tr_config,
    )

    ra = RetrievalAugmentation(config=ra_config, tree=str(index_path))
    logger.info(
        f"Đã tải thành công cây! Tổng số node: {len(ra.tree.all_nodes)}, "
        f"Số tầng: {ra.tree.num_layers}, Số lá: {len(ra.tree.leaf_nodes)}"
    )
    return ra


def select_index_file(index_dir: Path) -> Optional[Path]:
    """Tìm và cho phép người dùng chọn file index nếu có nhiều file."""
    if not index_dir.exists():
        return None

    # Bỏ qua các file *_raptor.pkl còn sót lại từ backend raptor_tree đã gỡ bỏ
    # (data/index/ không nằm trong git, nên máy đã ingest --raptor trước đây
    # vẫn còn file này trên đĩa) — nạp nó chắc chắn lỗi vì backend không còn nữa.
    pkl_files = [f for f in index_dir.glob("*.pkl") if not f.stem.endswith("_raptor")]
    if not pkl_files:
        return None

    if len(pkl_files) == 1:
        return pkl_files[0]

    print("\n📁 Tìm thấy các file index:")
    for idx, f in enumerate(pkl_files, 1):
        print(f"  [{idx}] {f.name}")

    while True:
        try:
            choice = input("\nChọn số thứ tự file index (hoặc Enter để chọn [1]): ").strip()
            if not choice:
                return pkl_files[0]
            choice_num = int(choice)
            if 1 <= choice_num <= len(pkl_files):
                return pkl_files[choice_num - 1]
            print("⚠️ Lựa chọn không hợp lệ, vui lòng thử lại.")
        except ValueError:
            print("⚠️ Vui lòng nhập số.")


async def ask_agent(graph, query: str, verbose: bool = False):
    """
    Gửi câu hỏi tới CRAG Agent, nhận câu trả lời, báo cáo token/chi phí, và
    trajectory (quỹ đạo thực thi: tool nào được gọi với args gì, lặp mấy
    vòng retrieve, feedback từng vòng, latency mỗi node, link LangSmith...).

    Chạy graph ĐÚNG MỘT LẦN qua TrajectoryRecorder (trước đây verbose=True
    chạy graph.astream() để in rồi graph.ainvoke() lần nữa để lấy đáp án —
    tốn gấp đôi token; hàm này gộp lại thành một lượt chạy duy nhất).

    Returns:
        (answer: str, token_report: str, trajectory: Trajectory)
    """
    from slidegen_rag.agent.models.llm import LLMModel
    from slidegen_rag.agent.observability.trajectory import TrajectoryRecorder, make_initial_state

    def _print_step(step) -> None:
        upd = step.update_summary
        print(f"  ▶ Node [{step.name}] hoàn tất ({step.duration_s:.2f}s).")
        if step.name == "retrieve" and step.llm_calls:
            tools_used = [tc for llm in step.llm_calls for tc in llm.tool_calls_emitted]
            print(f"     Tool calls: {tools_used}")
        elif step.name == "grade_documents" and "documents" in upd:
            print(f"     Tài liệu liên quan sau khi lọc: {upd['documents']}")
        elif step.name == "context_evaluator":
            print(f"     Đủ ngữ cảnh: {upd.get('context_sufficient')} | Feedback: {upd.get('feedback')}")

    if verbose:
        print("\n🔍 --- Bắt đầu quy trình Agent suy luận ---")

    recorder = TrajectoryRecorder(
        graph,
        tags=["cli"],
        metadata={"source": "main.py"},
        on_step=_print_step if verbose else None,
    )
    result, traj = await recorder.arun(make_initial_state(query), run_name=f"cli:{query[:60]}")
    answer = result.get("generation") or "Không có câu trả lời."

    if verbose:
        print("------------------------------------------\n")
        print(traj.render_text())
        print()

    report = LLMModel.get_usage_report()
    return answer, report, traj


async def ask_agentic(store, query: str, verbose: bool = False):
    """
    Same shape as ask_agent() (answer, token report, extra) but running the
    new agentic harness (search/find/open/summarize + bounded loop) instead
    of the CRAG graph. A fresh graph+registry is built per call — the
    ReferenceRegistry is per-run state (see harness/refs.py), never shared
    across questions.

    Not yet wrapped in TrajectoryRecorder (that integration — reinterpreting
    a CRAG "node boundary" as an agentic "iteration boundary" — is separate
    future work); this returns the compiled result state itself as the third
    element instead of a Trajectory, and token/cost still comes from the same
    LLMModel.get_usage_report() the CRAG path uses (the harness manually
    tracks usage from its bind_tools() calls — see harness/nodes.py:_track_usage).
    """
    from slidegen_rag.agent.harness.config import AgenticConfig
    from slidegen_rag.agent.harness.graph import build_agentic_graph, make_initial_state
    from slidegen_rag.agent.models.llm import LLMModel

    graph, registry = build_agentic_graph(store, AgenticConfig())
    result = await graph.ainvoke(make_initial_state(query), config={"recursion_limit": 60})
    answer = result.get("generation") or "Không có câu trả lời."

    if verbose:
        from langchain_core.messages import AIMessage
        tool_calls = [
            tc["name"] for m in result["messages"] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])
        ]
        print(f"\n🔍 --- Agentic harness: {result.get('iteration')} vòng lặp, tool calls: {tool_calls} ---")
        if result.get("forced_completion"):
            print("⚠️ Đã chạm giới hạn vòng lặp — câu trả lời là best-effort.")
        print()

    report = LLMModel.get_usage_report()
    return answer, report, result


async def interactive_loop(ask_fn, verbose: bool, agentic: bool = False):
    """Vòng lặp hỏi đáp tương tác trong terminal. `ask_fn(query, verbose)` is
    either `ask_agent` (bound to a compiled CRAG graph) or `ask_agentic`
    (bound to a DocStore) via functools.partial — see main()."""
    print("=" * 65)
    print(" 🤖 HỆ THỐNG HỎI ĐÁP THÔNG MINH (SLIDEGEN-RAG AGENT)")
    print(f"    - Chế độ: {'AGENTIC HARNESS' if agentic else 'CRAG (5-node)'}")
    print("    - Nhập câu hỏi và nhấn Enter để nhận câu trả lời.")
    print("    - Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print("    - Gõ 'verbose' để bật/tắt hiển thị chi tiết các bước Agent.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("\n👤 Bạn: ").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\nTạm biệt! 👋")
                break

            if query.lower() == "verbose":
                verbose = not verbose
                state_str = "BẬT" if verbose else "TẮT"
                print(f"ℹ️ Chế độ verbose đã được {state_str}.")
                continue

            print("\n⏳ Đang tìm kiếm và xử lý câu trả lời...")
            answer, report, _extra = await ask_fn(query, verbose=verbose)
            print(f"\n💡 Trợ lý:\n{answer}\n")
            print(f"{report}\n")
            print("-" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\n\nĐã dừng chương trình. Tạm biệt! 👋")
            break
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {e}", exc_info=True)


def main():
    from slidegen_rag.config import DATA_INDEX_DIR

    parser = argparse.ArgumentParser(
        description="Giao diện hỏi đáp tương tác với cây phân cấp tài liệu qua Agent."
    )
    parser.add_argument(
        "--index", "-i",
        type=str,
        default=None,
        help=f"Đường dẫn file .pkl index của cây. Mặc định tự động tìm trong {DATA_INDEX_DIR}",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Câu hỏi cụ thể để trả lời ngay (không vào chế độ tương tác).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị chi tiết các bước chạy của Agent (tools, grading, evaluator).",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Chỉ dùng cùng --query, và chỉ ở chế độ CRAG (không --agentic): ghi "
            "trajectory (quỹ đạo thực thi Agent — tool calls, số vòng retrieve, "
            "feedback, token/chi phí) ra file PATH."
        ),
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help=(
            "Dùng harness agentic mới (search/find/open/summarize + vòng lặp có "
            "biên, theo AgenticRAG) thay vì graph CRAG 5-node mặc định. Yêu cầu "
            "index schema_version >= 2 (xem scripts/migrate_index_v2.py)."
        ),
    )

    args = parser.parse_args()

    # 1. Xác định file index
    if args.index:
        index_path = Path(args.index)
    else:
        index_path = select_index_file(DATA_INDEX_DIR)
        if not index_path:
            logger.error(
                f"Không tìm thấy file .pkl phù hợp trong '{DATA_INDEX_DIR}'. "
                f"Hãy chạy 'uv run slidegen-ingest' trước để tạo index."
            )
            sys.exit(1)

    # 2. Khởi tạo RetrievalAugmentation
    try:
        ra = load_ra(index_path)
    except Exception as e:
        logger.error(f"Không thể khởi tạo RetrievalAugmentation: {e}", exc_info=True)
        sys.exit(1)

    if args.agentic:
        from slidegen_rag.agent.harness.doc_store import DocStore, DocStoreError

        try:
            store = DocStore.from_ra(ra)
        except DocStoreError as e:
            logger.error(
                f"Index chưa hỗ trợ harness agentic: {e} "
                f"Chạy 'uv run python scripts/migrate_index_v2.py --file {index_path}' trước."
            )
            sys.exit(1)
        logger.info("Agentic harness đã sẵn sàng!\n")
        ask_fn = lambda q, verbose: ask_agentic(store, q, verbose=verbose)  # noqa: E731
    else:
        from slidegen_rag.agent.graph.builder import build_graph
        from slidegen_rag.agent.tools.tools import set_retrieval_augmentation

        set_retrieval_augmentation(ra)
        logger.info("Đang khởi tạo Agent Graph...")
        graph = build_graph()
        logger.info("Agent đã sẵn sàng!\n")
        ask_fn = lambda q, verbose: ask_agent(graph, q, verbose=verbose)  # noqa: E731

    # 3. Thực thi câu hỏi hoặc mở interactive chat
    if args.query:
        answer, report, extra = asyncio.run(ask_fn(args.query, verbose=args.verbose))
        print(f"\n💡 Trợ lý:\n{answer}\n")
        print(f"{report}")
        if args.trajectory:
            if args.agentic:
                logger.warning("--trajectory chưa được hỗ trợ ở chế độ --agentic, bỏ qua.")
            else:
                traj_path = Path(args.trajectory)
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                traj_path.write_text(extra.render_text(), encoding="utf-8")
                logger.info(f"Đã ghi trajectory vào {traj_path}")
    else:
        asyncio.run(interactive_loop(ask_fn, verbose=args.verbose, agentic=args.agentic))


if __name__ == "__main__":
    main()
