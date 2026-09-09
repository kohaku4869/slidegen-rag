"""
ingest.py — CLI tool để index các file Markdown thành Tree và lưu xuống đĩa.

Usage:
    # Index tất cả file .md trong data/src/
    uv run python ingest.py

    # Index một file cụ thể
    uv run python ingest.py --file data/src/pldc.md

    # Chỉ định thư mục nguồn và đầu ra
    uv run python ingest.py --src data/src --out data/index

    # Ghi đè index đã tồn tại
    uv run python ingest.py --force

    # Dùng backend raptor_tree (GMM+UMAP clustering theo đúng paper RAPTOR)
    # thay vì backend tree mặc định (Markdown-heading). Lưu ra <stem>_raptor.pkl
    uv run python ingest.py --raptor --file data/src/pldc.md
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from tree.stats import BuildStats, format_ingest_report, format_multi_file_summary

# Fix Windows cp1252 encoding crash khi in UTF-8 ra stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Cau hinh logging ────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def _build_ra(cfg_tree, cfg_sum, use_raptor: bool = False):
    """
    Khởi tạo RetrievalAugmentation từ config.

    Lazy import để tránh load model khi chỉ chạy --help.
    use_raptor=True dùng backend `raptor_tree` (GMM+UMAP clustering theo đúng paper),
    mặc định (False) dùng backend `tree` (Markdown-heading, hành vi hiện tại).
    """
    if use_raptor:
        from raptor_tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
        from raptor_tree.builder.tree_builder import TreeBuilderConfig
        from raptor_tree.models.EmbeddingModels import EmbeddingModel
        from raptor_tree.models.SummarizationModels import ChatNVIDIASummarizationModel
        from raptor_tree.retriever.tree_retriever import TreeRetrieverConfig
    else:
        from tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
        from tree.builder.tree_builder import TreeBuilderConfig
        from tree.models.EmbeddingModels import EmbeddingModel
        from tree.models.SummarizationModels import ChatNVIDIASummarizationModel
        from tree.retriever.tree_retriever import TreeRetrieverConfig

    embedding_model = EmbeddingModel(
        model_name=cfg_tree.EMBEDDING_MODEL_NAME,
        base_url=getattr(cfg_tree, "EMBEDDING_BASE_URL", None),
    )
    summarization_model = ChatNVIDIASummarizationModel(
        model=cfg_sum.MODEL,
        fallback_model=getattr(cfg_sum, "FALLBACK_MODEL", "gemini-2.5-flash"),
        temperature=cfg_sum.TEMPERATURE,
    )

    tb_kwargs = dict(
        summarization_length=cfg_tree.SUMMARIZATION_LENGTH,
        summarization_model=summarization_model,
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

    from config import RetrieverConfig
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

    return RetrievalAugmentation(config=ra_config)


def ingest_file(
    md_path: Path, out_dir: Path, force: bool, cfg_tree, cfg_sum, use_raptor: bool = False
) -> Tuple[bool, Optional[BuildStats]]:
    """
    Index một file Markdown thành Tree và lưu file .pkl, đồng thời xuất báo cáo hiệu năng.

    Args:
        md_path: Đường dẫn tới file .md nguồn.
        out_dir: Thư mục lưu file .pkl đầu ra.
        force:   Ghi đè nếu file .pkl đã tồn tại.
        cfg_tree: TreeConfig hoặc RaptorTreeConfig instance (tùy use_raptor).
        cfg_sum:  SummarizationConfig instance.
        use_raptor: Dùng backend `raptor_tree` (GMM+UMAP) thay vì `tree` (Markdown-heading).

    Returns:
        (True, BuildStats) nếu thành công, (False, None) nếu bỏ qua hoặc lỗi.
    """
    suffix = "_raptor" if use_raptor else ""
    pkl_path = out_dir / (md_path.stem + suffix + ".pkl")

    if pkl_path.exists() and not force:
        logger.info(f"[SKIP] Index đã tồn tại: {pkl_path} (dùng --force để ghi đè)")
        return False, None

    backend_name = "raptor_tree (GMM+UMAP)" if use_raptor else "tree (Markdown-heading)"
    logger.info(f"[START] Đang index: {md_path.name} [backend={backend_name}]")
    t_file_start = time.perf_counter()

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            logger.warning(f"[WARN] File rỗng, bỏ qua: {md_path.name}")
            return False, None

        ra = _build_ra(cfg_tree, cfg_sum, use_raptor=use_raptor)

        # Bỏ qua warning "Overwriting existing tree" — đây là RA mới hoàn toàn
        ra.tree = None
        ra.tree = ra.tree_builder.build_from_text(markdown_text=text)

        if use_raptor:
            from raptor_tree.retriever.tree_retriever import TreeRetriever
        else:
            from tree.retriever.tree_retriever import TreeRetriever
        ra.retriever = TreeRetriever(ra.tree_retriever_config, ra.tree)

        t_save_start = time.perf_counter()
        ra.save(str(pkl_path))
        save_time = time.perf_counter() - t_save_start
        total_file_time = time.perf_counter() - t_file_start
        file_size_bytes = pkl_path.stat().st_size if pkl_path.exists() else 0

        logger.info(f"[DONE] Đã lưu index: {pkl_path}")

        # Cập nhật và in báo cáo thống kê hiệu năng chi tiết
        stats: Optional[BuildStats] = getattr(ra.tree, "build_stats", None)
        if stats is not None and isinstance(stats, BuildStats):
            stats.filename = md_path.name
            stats.src_path = str(md_path)
            stats.out_path = str(pkl_path)
            stats.save_time = save_time
            stats.total_time = total_file_time
            stats.file_size_bytes = file_size_bytes

            report = format_ingest_report(stats)
            print("\n" + report + "\n", flush=True)

        return True, stats

    except Exception as e:
        logger.error(f"[ERROR] Lỗi khi index {md_path.name}: {e}", exc_info=True)
        return False, None


def main():
    from config import DATA_SRC_DIR, DATA_INDEX_DIR, TreeConfig, RaptorTreeConfig, SummarizationConfig

    parser = argparse.ArgumentParser(
        description="Index các file Markdown thành RAPTOR Tree và lưu xuống đĩa."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Đường dẫn tới file .md cụ thể cần index. Nếu không truyền, index toàn bộ --src.",
    )
    parser.add_argument(
        "--src", "-s",
        type=str,
        default=str(DATA_SRC_DIR),
        help=f"Thư mục chứa các file .md nguồn. Mặc định: {DATA_SRC_DIR}",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default=str(DATA_INDEX_DIR),
        help=f"Thư mục lưu file .pkl. Mặc định: {DATA_INDEX_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè file index đã tồn tại.",
    )
    parser.add_argument(
        "--raptor",
        action="store_true",
        help=(
            "Dùng backend raptor_tree (GMM+UMAP clustering theo đúng paper RAPTOR) "
            "thay vì backend tree mặc định (Markdown-heading). "
            "Index được lưu với hậu tố '_raptor.pkl'."
        ),
    )

    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_tree = RaptorTreeConfig() if args.raptor else TreeConfig()
    cfg_sum = SummarizationConfig()

    # ── Xác định danh sách file cần index ───────────────────────────────────
    if args.file:
        md_files = [Path(args.file)]
        if not md_files[0].exists():
            logger.error(f"File không tồn tại: {md_files[0]}")
            sys.exit(1)
    else:
        md_files = sorted(src_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"Không tìm thấy file .md nào trong: {src_dir}")
            sys.exit(0)

    logger.info(f"Tìm thấy {len(md_files)} file(s) cần index.")
    logger.info(f"  Nguồn  : {src_dir}")
    logger.info(f"  Đầu ra : {out_dir}")
    logger.info(f"  Force  : {args.force}")
    logger.info(f"  Backend: {'raptor_tree (GMM+UMAP)' if args.raptor else 'tree (Markdown-heading)'}")
    logger.info("-" * 60)

    # ── Index từng file ──────────────────────────────────────────────────────
    success, skipped, failed = 0, 0, 0
    all_stats: list[BuildStats] = []
    out_suffix = "_raptor" if args.raptor else ""

    for md_path in md_files:
        ok, stats = ingest_file(md_path, out_dir, args.force, cfg_tree, cfg_sum, use_raptor=args.raptor)
        if ok:
            success += 1
            if stats:
                all_stats.append(stats)
        elif not args.force and (out_dir / (md_path.stem + out_suffix + ".pkl")).exists():
            skipped += 1
        else:
            failed += 1

    # In bảng tổng hợp nếu index nhiều hơn 1 file
    if len(all_stats) > 1:
        multi_summary = format_multi_file_summary(all_stats)
        print(multi_summary + "\n", flush=True)

    logger.info("-" * 60)
    logger.info(
        f"Hoàn tất: {success} thành công | {skipped} bỏ qua | {failed} lỗi"
    )


if __name__ == "__main__":
    main()

