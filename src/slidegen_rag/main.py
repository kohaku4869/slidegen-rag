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

    # 5. Quản lý phiên chat (cần DATABASE_URL, xem memory/ — xem AGENTS.md)
    uv run slidegen-ask --list-sessions
    uv run slidegen-ask --resume                # phiên active gần nhất của tài liệu đang nạp
    uv run slidegen-ask --session <uuid>        # resume đúng phiên đó
    uv run slidegen-ask --no-memory             # tắt persist cho lần chạy này
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

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
    from slidegen_rag.config import RetrieverConfig, TreeConfig
    from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
    from slidegen_rag.tree.RetrievalAugmentation import (
        RetrievalAugmentation,
        RetrievalAugmentationConfig,
    )
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
        embedding_model=embedding_model,
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


def select_index_file(index_dir: Path) -> Path | None:
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
            choice = input(
                "\nChọn số thứ tự file index (hoặc Enter để chọn [1]): "
            ).strip()
            if not choice:
                return pkl_files[0]
            choice_num = int(choice)
            if 1 <= choice_num <= len(pkl_files):
                return pkl_files[choice_num - 1]
            print("⚠️ Lựa chọn không hợp lệ, vui lòng thử lại.")
        except ValueError:
            print("⚠️ Vui lòng nhập số.")


async def ask_agent(
    graph, query: str, verbose: bool = False, drill_candidates: list | None = None
):
    """
    Gửi câu hỏi tới CRAG Agent, nhận câu trả lời, báo cáo token/chi phí, và
    trajectory (quỹ đạo thực thi: tool nào được gọi với args gì, lặp mấy
    vòng retrieve, feedback từng vòng, latency mỗi node, link LangSmith...).

    Chạy graph ĐÚNG MỘT LẦN qua TrajectoryRecorder (trước đây verbose=True
    chạy graph.astream() để in rồi graph.ainvoke() lần nữa để lấy đáp án —
    tốn gấp đôi token; hàm này gộp lại thành một lượt chạy duy nhất).

    Args:
        drill_candidates: Seed ban đầu cho state["drill_candidates"] — dùng
            bởi interactive_loop() khi câu hỏi là follow-up của lượt trước
            (xem agent/graph/conversation.py::ConversationManager).

    Returns:
        (answer: str, token_report: str, trajectory: Trajectory)
    """
    from slidegen_rag.agent.graph.state import make_initial_state
    from slidegen_rag.agent.models.llm import LLMModel
    from slidegen_rag.agent.observability.trajectory import TrajectoryRecorder

    def _print_step(step) -> None:
        upd = step.update_summary
        print(f"  ▶ Node [{step.name}] hoàn tất ({step.duration_s:.2f}s).")
        if step.name == "retrieve" and step.llm_calls:
            tools_used = [tc for llm in step.llm_calls for tc in llm.tool_calls_emitted]
            print(f"     Tool calls: {tools_used}")
        elif step.name == "grade_documents" and "documents" in upd:
            print(f"     Tài liệu liên quan sau khi lọc: {upd['documents']}")
        elif step.name == "context_evaluator":
            print(
                f"     Đủ ngữ cảnh: {upd.get('context_sufficient')} | Feedback: {upd.get('feedback')}"
            )

    if verbose:
        print("\n🔍 --- Bắt đầu quy trình Agent suy luận ---")

    recorder = TrajectoryRecorder(
        graph,
        tags=["cli"],
        metadata={"source": "main.py"},
        on_step=_print_step if verbose else None,
    )
    result, traj = await recorder.arun(
        make_initial_state(query, drill_candidates=drill_candidates),
        run_name=f"cli:{query[:60]}",
    )
    answer = result.get("generation") or "Không có câu trả lời."

    if verbose:
        print("------------------------------------------\n")
        print(traj.render_text())
        print()

    report = LLMModel.get_usage_report()
    return answer, report, traj


async def _build_repo():
    """Dựng `PostgresSessionRepository`, fail loudly (sys.exit) nếu
    `DATABASE_URL` đã khai nhưng không kết nối được — xem
    memory/engine.py::ping() và plan4.md B4 (memory KHÔNG theo quy ước
    fail-soft của observability trong repo này)."""
    from slidegen_rag.memory.engine import get_sessionmaker, ping
    from slidegen_rag.memory.repository import PostgresSessionRepository

    try:
        await ping()
    except Exception as e:
        logger.error(
            f"DATABASE_URL đã khai nhưng không kết nối được Postgres: {e}\n"
            f"Sửa DATABASE_URL trong .env, hoặc chạy lại với --no-memory."
        )
        sys.exit(1)
    return PostgresSessionRepository(get_sessionmaker())


async def _print_sessions() -> None:
    """`--list-sessions`: in bảng phiên đã lưu rồi thoát — không load tree,
    không build agent graph."""
    from slidegen_rag.config import MemoryConfig

    if not MemoryConfig.DATABASE_URL:
        print("DATABASE_URL chưa được cấu hình — chưa có phiên chat nào được lưu.")
        return
    repo = await _build_repo()
    rows = await repo.list_sessions(limit=50)
    if not rows:
        print("Chưa có phiên chat nào được lưu.")
        return
    print(f"{'SESSION':38}  {'TRẠNG THÁI':10}  {'TÀI LIỆU':20}  HOẠT ĐỘNG GẦN NHẤT")
    for r in rows:
        doc = (r.bound_doc_id or "-")[:20]
        print(
            f"{r.id!s:38}  {r.status:10}  {doc:20}  {r.last_active_at:%Y-%m-%d %H:%M}"
        )


async def _resolve_conversation(
    args, index_path: Path, doc_ref, *, always_create: bool = False
):
    """Dựng `ConversationManager` theo cờ CLI (xem plan4.md B5).

    Mặc định KHÔNG cờ nào ở chế độ interactive (`always_create=True`): tạo
    phiên MỚI + persist, KHÔNG auto-resume — auto-resume sẽ âm thầm đổi UX vì
    `contextualize` viết lại câu hỏi đầu tiên dựa trên lịch sử cũ mà người
    dùng không mong đợi.

    Chế độ `--query` một lần (`always_create=False`) chỉ gọi hàm này khi có
    `--session`/`--resume` — giữ bảng `sessions` sạch khỏi các lần chạy
    benchmark/eval/trajectory.
    """
    from slidegen_rag.agent.graph.conversation import ConversationManager
    from slidegen_rag.config import MemoryConfig

    if args.no_memory or not MemoryConfig.ENABLED:
        if always_create:
            return ConversationManager(max_turns=MemoryConfig.SESSION_MAX_TURNS)
        return None

    repo = await _build_repo()

    if args.session:
        try:
            session_id = UUID(args.session)
        except ValueError:
            logger.error(f"--session không phải UUID hợp lệ: {args.session}")
            sys.exit(1)
        try:
            conv = await ConversationManager.resume(
                repo, session_id, doc_ref, max_turns=MemoryConfig.SESSION_MAX_TURNS
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        print(f"📌 Session: {conv.session_id}  (đã resume)")
        return conv

    if args.resume and not args.new_session:
        state = await repo.latest_session(bound_doc_id=doc_ref.doc_id)
        if state is not None:
            conv = await ConversationManager.resume(
                repo, state.id, doc_ref, max_turns=MemoryConfig.SESSION_MAX_TURNS
            )
            print(f"📌 Session: {conv.session_id}  (đã resume phiên gần nhất)")
            return conv
        logger.info("Không có phiên active nào cho tài liệu này — tạo phiên mới.")

    state = await repo.create_session(
        bound_doc_id=doc_ref.doc_id,
        index_path=str(index_path),
        max_turns=MemoryConfig.SESSION_MAX_TURNS,
    )
    conv = ConversationManager(
        max_turns=state.max_turns, repo=repo, session_id=state.id, doc_ref=doc_ref
    )
    print(f"📌 Session: {conv.session_id}  (tiếp tục: uv run slidegen-ask --resume)")
    return conv


async def _async_entry(args, ask_fn, index_path: Path, doc_ref) -> None:
    """Điểm vào async duy nhất sau khi tree/graph đã sẵn sàng — cả nhánh
    one-shot (`--query`) lẫn interactive chạy trong CÙNG một event loop, vì
    engine SQLAlchemy async (asyncpg) bị buộc vào loop đã tạo ra nó; tách
    thành hai `asyncio.run()` riêng sẽ dùng lại engine qua loop đã đóng."""
    if args.query:
        answer, report, extra = await ask_fn(args.query, verbose=args.verbose)
        print(f"\n💡 Trợ lý:\n{answer}\n")
        print(f"{report}")
        if args.trajectory:
            traj_path = Path(args.trajectory)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_path.write_text(extra.render_text(), encoding="utf-8")
            logger.info(f"Đã ghi trajectory vào {traj_path}")
        if args.session or args.resume:
            conv = await _resolve_conversation(args, index_path, doc_ref)
            if conv is not None:
                document_node_ids = extra.final_state.get("document_node_ids", [])
                await conv.arecord_turn(
                    args.query, args.query, answer, document_node_ids
                )
                print(f"📌 Đã lưu vào session {conv.session_id}")
        return

    conv = await _resolve_conversation(args, index_path, doc_ref, always_create=True)
    await interactive_loop(ask_fn, verbose=args.verbose, conv=conv)


async def interactive_loop(ask_fn, verbose: bool, conv=None):
    """Vòng lặp hỏi đáp tương tác trong terminal. `ask_fn(query, verbose)` là
    `ask_agent` đã bind sẵn graph CRAG đã compile — xem main().

    `conv` (ConversationManager) quản lý hội thoại nhiều lượt: viết lại câu
    hỏi follow-up thành câu độc lập, seed drill_candidates từ node đã trích
    dẫn ở lượt trước, và nén lịch sử khi quá dài — xem
    agent/graph/conversation.py."""
    print("=" * 65)
    print(" 🤖 HỆ THỐNG HỎI ĐÁP THÔNG MINH (SLIDEGEN-RAG AGENT)")
    print("    - Chế độ: CRAG (5-node)")
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
            if conv is not None:
                standalone_query = await conv.contextualize(query)
                if standalone_query != query:
                    print(f"   (Đã diễn giải lại: {standalone_query})")
                answer, report, extra = await ask_fn(
                    standalone_query,
                    verbose=verbose,
                    drill_candidates=conv.seed_drill_candidates(),
                )
                document_node_ids = extra.final_state.get("document_node_ids", [])
                await conv.arecord_turn(
                    query, standalone_query, answer, document_node_ids
                )
                await conv.maybe_compact()
            else:
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
    from slidegen_rag.config import DATA_INDEX_DIR, MODEL_CHOICES, apply_model_choice

    parser = argparse.ArgumentParser(
        description="Giao diện hỏi đáp tương tác với cây phân cấp tài liệu qua Agent."
    )
    parser.add_argument(
        "--index",
        "-i",
        type=str,
        default=None,
        help=f"Đường dẫn file .pkl index của cây. Mặc định tự động tìm trong {DATA_INDEX_DIR}",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Câu hỏi cụ thể để trả lời ngay (không vào chế độ tương tác).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Hiển thị chi tiết các bước chạy của Agent (tools, grading, evaluator).",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Chỉ dùng cùng --query: ghi trajectory (quỹ đạo thực thi Agent — tool "
            "calls, số vòng retrieve, feedback, token/chi phí) ra file PATH."
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="UUID",
        help="Resume đúng phiên chat có UUID này.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume phiên active gần nhất của tài liệu đang nạp.",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Luôn tạo phiên chat mới, kể cả khi có --resume (ghi đè --resume).",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="In danh sách phiên chat đã lưu rồi thoát (không load tài liệu).",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Tắt persist session memory (Postgres) cho lần chạy này.",
    )

    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="gemini",
        help="LLM dùng cho Agent: gemini (mặc định) hoặc qwen (qwen/qwen3.7-flash qua OpenRouter).",
    )

    args = parser.parse_args()
    apply_model_choice(args.model)

    if args.list_sessions:
        asyncio.run(_print_sessions())
        return

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

    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.tools.tools import set_retrieval_augmentation
    from slidegen_rag.memory.dto import DocumentRef

    set_retrieval_augmentation(ra)
    logger.info("Đang khởi tạo Agent Graph...")
    graph = build_graph()
    logger.info("Agent đã sẵn sàng!\n")
    ask_fn = lambda q, verbose=False, drill_candidates=None: ask_agent(
        graph, q, verbose=verbose, drill_candidates=drill_candidates
    )
    doc_ref = DocumentRef(
        doc_id=ra.tree.doc_id or index_path.stem, source_sha1=ra.tree.source_sha1
    )

    # 3. Thực thi câu hỏi hoặc mở interactive chat — CÙNG một event loop cho
    # cả agent graph lẫn memory (xem _async_entry).
    asyncio.run(_async_entry(args, ask_fn, index_path, doc_ref))


if __name__ == "__main__":
    main()
