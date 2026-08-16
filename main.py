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
    """
    Khởi tạo RetrievalAugmentation và load cây phân cấp từ file .pkl.
    """
    from config import TreeConfig, RetrieverConfig
    from tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from tree.builder.tree_builder import TreeBuilderConfig
    from tree.models.EmbeddingModels import EmbeddingModel
    from tree.retriever.tree_retriever import TreeRetrieverConfig

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file index tại: {index_path}")

    logger.info(f"Đang tải cây từ: {index_path}")
    embedding_model = EmbeddingModel(model_name=TreeConfig.EMBEDDING_MODEL_NAME)

    tb_config = TreeBuilderConfig(
        summarization_length=TreeConfig.SUMMARIZATION_LENGTH,
        embedding_models={TreeConfig.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=TreeConfig.CLUSTER_EMBEDDING_MODEL,
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

    pkl_files = list(index_dir.glob("*.pkl"))
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


async def ask_agent(graph, query: str, verbose: bool = False) -> tuple[str, str]:
    """
    Gửi câu hỏi tới CRAG Agent, nhận câu trả lời và báo cáo token/chi phí.
    
    Returns:
        (answer: str, token_report: str)
    """
    from agent.models.llm import LLMModel

    # Reset bộ đếm token cho lượt hỏi mới
    LLMModel.reset_usage()

    initial_state = {
        "query": query,
        "tool_calls": [],
        "documents": [],
        "feedback": "",
        "context_sufficient": False,
        "generation": "",
        "retrieve_count": 0,
    }

    if verbose:
        print("\n🔍 --- Bắt đầu quy trình Agent suy luận ---")
        async for output in graph.astream(initial_state):
            for node_name, node_output in output.items():
                print(f"  ▶ Node [{node_name}] hoàn tất.")
                if node_name == "retrieve" and "tool_calls" in node_output:
                    tools_used = [tc.get("name") for tc in node_output["tool_calls"]]
                    print(f"     Tool calls: {tools_used}")
                elif node_name == "grade_documents" and "documents" in node_output:
                    print(f"     Tài liệu liên quan sau khi lọc: {len(node_output['documents'])}")
                elif node_name == "context_evaluator":
                    suff = node_output.get("context_sufficient")
                    fb = node_output.get("feedback")
                    print(f"     Đủ ngữ cảnh: {suff} | Feedback: {fb}")
        print("------------------------------------------\n")
        result = await graph.ainvoke(initial_state)
        answer = result.get("generation", "Không có câu trả lời.")
    else:
        result = await graph.ainvoke(initial_state)
        answer = result.get("generation", "Không có câu trả lời.")

    report = LLMModel.get_usage_report()
    return answer, report


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
            answer, report = await ask_agent(graph, query, verbose=verbose)
            print(f"\n💡 Trợ lý:\n{answer}\n")
            print(f"{report}\n")
            print("-" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\n\nĐã dừng chương trình. Tạm biệt! 👋")
            break
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {e}", exc_info=True)


def main():
    from config import DATA_INDEX_DIR

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

    args = parser.parse_args()

    from agent.graph.builder import build_graph
    from agent.tools.tools import set_retrieval_augmentation

    # 1. Xác định file index
    if args.index:
        index_path = Path(args.index)
    else:
        index_path = select_index_file(DATA_INDEX_DIR)
        if not index_path:
            logger.error(
                f"Không tìm thấy file .pkl nào trong '{DATA_INDEX_DIR}'. "
                f"Hãy chạy 'uv run python ingest.py' trước để tạo index."
            )
            sys.exit(1)

    # 2. Khởi tạo RetrievalAugmentation & inject vào Tool
    try:
        ra = load_ra(index_path)
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
        answer, report = asyncio.run(ask_agent(graph, args.query, verbose=args.verbose))
        print(f"\n💡 Trợ lý:\n{answer}\n")
        print(f"{report}")
    else:
        asyncio.run(interactive_loop(graph, verbose=args.verbose))


if __name__ == "__main__":
    main()
