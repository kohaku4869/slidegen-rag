"""
main.py — Giao diện Hỏi Đáp (Q&A) tương tác sử dụng Corrective RAG Agent và RAPTOR Tree.

Cách sử dụng:
    # 1. Chế độ chat tương tác với file index mặc định (hoặc chọn từ danh sách)
    uv run python main.py

    # 2. Chỉ định file index cụ thể
    uv run python main.py --index data/index/pldc.pkl

    # 3. Hỏi trực tiếp một câu không cần vào vòng lặp chat
    uv run python main.py --query "Quy phạm pháp luật là gì?"

    # 4. Bật chế độ chi tiết (xem các bước Agent suy luận và gọi tools)
    uv run python main.py --verbose

    # 5. Dùng backend raptor_tree (GMM+UMAP clustering theo đúng paper RAPTOR)
    #    thay vì backend tree mặc định (Markdown-heading). Cần index được tạo
    #    bằng 'uv run python ingest.py --raptor ...' trước (file *_raptor.pkl)
    uv run python main.py --raptor --index data/index/pldc_raptor.pkl
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


def load_ra(index_path: Path, use_raptor: bool = False):
    """
    Khởi tạo RetrievalAugmentation và load cây phân cấp từ file .pkl.

    use_raptor=True dùng backend `raptor_tree` (GMM+UMAP clustering theo đúng paper),
    mặc định (False) dùng backend `tree` (Markdown-heading, hành vi hiện tại).
    Lưu ý: file .pkl phải được tạo bằng đúng backend tương ứng (xem ingest.py --raptor).
    """
    from slidegen_rag.config import TreeConfig, RaptorTreeConfig, RetrieverConfig

    if use_raptor:
        from slidegen_rag.raptor_tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
        from slidegen_rag.raptor_tree.builder.tree_builder import TreeBuilderConfig
        from slidegen_rag.raptor_tree.models.EmbeddingModels import EmbeddingModel
        from slidegen_rag.raptor_tree.retriever.tree_retriever import TreeRetrieverConfig
        cfg_tree = RaptorTreeConfig
    else:
        from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
        from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
        from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
        from slidegen_rag.tree.retriever.tree_retriever import TreeRetrieverConfig
        cfg_tree = TreeConfig

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file index tại: {index_path}")

    backend_name = "raptor_tree (GMM+UMAP)" if use_raptor else "tree (Markdown-heading)"
    logger.info(f"Đang tải cây từ: {index_path} [backend={backend_name}]")
    embedding_model = EmbeddingModel(model_name=cfg_tree.EMBEDDING_MODEL_NAME)

    tb_kwargs = dict(
        summarization_length=cfg_tree.SUMMARIZATION_LENGTH,
        embedding_models={cfg_tree.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=cfg_tree.CLUSTER_EMBEDDING_MODEL,
    )
    if use_raptor:
        tb_kwargs.update(
            max_tokens=cfg_tree.MAX_TOKENS,
            chunk_overlap=cfg_tree.CHUNK_OVERLAP,
            num_layers=cfg_tree.NUM_LAYERS,
            reduction_dimension=cfg_tree.REDUCTION_DIMENSION,
            cluster_threshold=cfg_tree.CLUSTER_THRESHOLD,
            max_clusters=cfg_tree.MAX_CLUSTERS,
            max_length_in_cluster=cfg_tree.MAX_LENGTH_IN_CLUSTER,
        )
    tb_config = TreeBuilderConfig(**tb_kwargs)

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


def select_index_file(index_dir: Path, use_raptor: bool = False) -> Optional[Path]:
    """
    Tìm và cho phép người dùng chọn file index nếu có nhiều file.

    Khi use_raptor=True, chỉ tìm các file '*_raptor.pkl' (tạo bởi 'ingest.py --raptor')
    để tránh nạp nhầm index của backend `tree` (Markdown-heading) vào backend `raptor_tree`.
    """
    if not index_dir.exists():
        return None

    pkl_files = list(index_dir.glob("*_raptor.pkl")) if use_raptor else [
        f for f in index_dir.glob("*.pkl") if not f.stem.endswith("_raptor")
    ]
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


async def interactive_loop(graph, verbose: bool):
    """Vòng lặp hỏi đáp tương tác trong terminal."""
    print("=" * 65)
    print(" 🤖 HỆ THỐNG HỎI ĐÁP THÔNG MINH (SLIDEGEN-RAG AGENT)")
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
            answer, report, _traj = await ask_agent(graph, query, verbose=verbose)
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
        description="Giao diện hỏi đáp tương tác với RAPTOR Tree qua Agent."
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
        "--raptor",
        action="store_true",
        help=(
            "Dùng backend raptor_tree (GMM+UMAP clustering theo đúng paper RAPTOR) "
            "thay vì backend tree mặc định (Markdown-heading). "
            "Cần index dạng '*_raptor.pkl' được tạo bằng 'ingest.py --raptor'."
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Chỉ dùng cùng --query: ghi trajectory (quỹ đạo thực thi Agent — "
            "tool calls, số vòng retrieve, feedback, token/chi phí) ra file PATH."
        ),
    )

    args = parser.parse_args()

    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.tools.tools import set_retrieval_augmentation

    # 1. Xác định file index
    if args.index:
        index_path = Path(args.index)
    else:
        index_path = select_index_file(DATA_INDEX_DIR, use_raptor=args.raptor)
        if not index_path:
            backend_hint = "ingest.py --raptor" if args.raptor else "ingest.py"
            logger.error(
                f"Không tìm thấy file .pkl phù hợp trong '{DATA_INDEX_DIR}'. "
                f"Hãy chạy 'uv run python {backend_hint}' trước để tạo index."
            )
            sys.exit(1)

    # 2. Khởi tạo RetrievalAugmentation & inject vào Tool
    try:
        ra = load_ra(index_path, use_raptor=args.raptor)
        set_retrieval_augmentation(ra)
    except Exception as e:
        logger.error(f"Không thể khởi tạo RetrievalAugmentation: {e}", exc_info=True)
        sys.exit(1)

    # 3. Khởi tạo Agent Graph
    logger.info("Đang khởi tạo Agent Graph...")
    graph = build_graph()
    logger.info("Agent đã sẵn sàng!\n")

    # 4. Thực thi câu hỏi hoặc mở interactive chat
    if args.query:
        answer, report, traj = asyncio.run(ask_agent(graph, args.query, verbose=args.verbose))
        print(f"\n💡 Trợ lý:\n{answer}\n")
        print(f"{report}")
        if args.trajectory:
            traj_path = Path(args.trajectory)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_path.write_text(traj.render_text(), encoding="utf-8")
            logger.info(f"Đã ghi trajectory vào {traj_path}")
    else:
        asyncio.run(interactive_loop(graph, verbose=args.verbose))


if __name__ == "__main__":
    main()
