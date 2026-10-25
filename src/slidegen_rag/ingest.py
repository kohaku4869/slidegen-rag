"""
ingest.py — CLI tool để index các file Markdown thành Tree và lưu xuống đĩa.

Usage:
    # Index tất cả file .md trong data/src/
    uv run slidegen-ingest

    # Index một file cụ thể
    uv run slidegen-ingest --file data/src/pldc.md

    # Chỉ định thư mục nguồn và đầu ra
    uv run slidegen-ingest --src data/src --out data/index

    # Ghi đè index đã tồn tại
    uv run slidegen-ingest --force
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from slidegen_rag.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    EmptyDocumentError,
    load_file,
)
from slidegen_rag.tree.stats import BuildStats, format_ingest_report, format_multi_file_summary

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


def _build_ra(cfg_tree, cfg_sum):
    """Khởi tạo RetrievalAugmentation từ config. Lazy import để tránh load model khi chỉ chạy --help."""
    from slidegen_rag.tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from slidegen_rag.tree.builder.tree_builder import TreeBuilderConfig
    from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
    from slidegen_rag.tree.models.SummarizationModels import ChatNVIDIASummarizationModel
    from slidegen_rag.tree.retriever.tree_retriever import TreeRetrieverConfig

    embedding_model = EmbeddingModel(
        model_name=cfg_tree.EMBEDDING_MODEL_NAME,
        base_url=getattr(cfg_tree, "EMBEDDING_BASE_URL", None),
    )
    summarization_model = ChatNVIDIASummarizationModel(
        model=cfg_sum.MODEL,
        fallback_model=getattr(cfg_sum, "FALLBACK_MODEL", "gemini-2.5-flash"),
        temperature=cfg_sum.TEMPERATURE,
    )

    tb_config = TreeBuilderConfig(
        summarization_length=cfg_tree.SUMMARIZATION_LENGTH,
        summarization_model=summarization_model,
        embedding_models={cfg_tree.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=cfg_tree.CLUSTER_EMBEDDING_MODEL,
    )

    from slidegen_rag.config import RetrieverConfig
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

    return RetrievalAugmentation(config=ra_config)


def ingest_file(
    md_path: Path, out_dir: Path, force: bool, cfg_tree, cfg_sum
) -> Tuple[bool, Optional[BuildStats]]:
    """
    Index một file Markdown thành Tree và lưu file .pkl, đồng thời xuất báo cáo hiệu năng.

    Args:
        md_path: Đường dẫn tới file .md nguồn.
        out_dir: Thư mục lưu file .pkl đầu ra.
        force:   Ghi đè nếu file .pkl đã tồn tại.
        cfg_tree: TreeConfig instance.
        cfg_sum:  SummarizationConfig instance.

    Returns:
        (True, BuildStats) nếu thành công, (False, None) nếu bỏ qua hoặc lỗi.
    """
    pkl_path = out_dir / (md_path.stem + ".pkl")

    if pkl_path.exists() and not force:
        logger.info(f"[SKIP] Index đã tồn tại: {pkl_path} (dùng --force để ghi đè)")
        return False, None

    logger.info(f"[START] Đang index: {md_path.name}")
    t_file_start = time.perf_counter()

    try:
        try:
            text = load_file(md_path)
        except EmptyDocumentError as e:
            logger.warning(f"[WARN] {e}, bỏ qua")
            return False, None

        ra = _build_ra(cfg_tree, cfg_sum)

        # Bỏ qua warning "Overwriting existing tree" — đây là RA mới hoàn toàn
        ra.tree = None
        ra.tree = ra.tree_builder.build_from_text(
            markdown_text=text, doc_id=md_path.stem, source_path=str(md_path)
        )

        from slidegen_rag.tree.retriever.tree_retriever import TreeRetriever
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
    from slidegen_rag.config import DATA_SRC_DIR, DATA_INDEX_DIR, TreeConfig, SummarizationConfig

    parser = argparse.ArgumentParser(
        description="Index các file Markdown/DOCX/PDF thành cây phân cấp và lưu xuống đĩa."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Đường dẫn tới file cụ thể (.md/.txt/.docx/.pdf) cần index. Nếu không truyền, index toàn bộ --src.",
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

    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_tree = TreeConfig()
    cfg_sum = SummarizationConfig()

    # ── Xác định danh sách file cần index ───────────────────────────────────
    if args.file:
        md_files = [Path(args.file)]
        if not md_files[0].exists():
            logger.error(f"File không tồn tại: {md_files[0]}")
            sys.exit(1)
    else:
        md_files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not md_files:
            logger.warning(f"Không tìm thấy file nguồn ({', '.join(SUPPORTED_EXTENSIONS)}) nào trong: {src_dir}")
            sys.exit(0)

    logger.info(f"Tìm thấy {len(md_files)} file(s) cần index.")
    logger.info(f"  Nguồn  : {src_dir}")
    logger.info(f"  Đầu ra : {out_dir}")
    logger.info(f"  Force  : {args.force}")
    logger.info("-" * 60)

    # ── Index từng file ──────────────────────────────────────────────────────
    success, skipped, failed = 0, 0, 0
    all_stats: list[BuildStats] = []

    for md_path in md_files:
        ok, stats = ingest_file(md_path, out_dir, args.force, cfg_tree, cfg_sum)
        if ok:
            success += 1
            if stats:
                all_stats.append(stats)
        elif not args.force and (out_dir / (md_path.stem + ".pkl")).exists():
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
