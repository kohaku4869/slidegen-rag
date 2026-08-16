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
"""

import argparse
import logging
import sys
from pathlib import Path

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
    """
    Khởi tạo RetrievalAugmentation từ config.

    Lazy import để tránh load model khi chỉ chạy --help.
    """
    from tree.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
    from tree.builder.tree_builder import TreeBuilderConfig
    from tree.models.EmbeddingModels import EmbeddingModel
    from tree.models.SummarizationModels import ChatNVIDIASummarizationModel
    from tree.retriever.tree_retriever import TreeRetrieverConfig

    embedding_model = EmbeddingModel(model_name=cfg_tree.EMBEDDING_MODEL_NAME)
    summarization_model = ChatNVIDIASummarizationModel(
        model=cfg_sum.MODEL,
        temperature=cfg_sum.TEMPERATURE,
    )

    tb_config = TreeBuilderConfig(
        summarization_length=cfg_tree.SUMMARIZATION_LENGTH,
        summarization_model=summarization_model,
        embedding_models={cfg_tree.EMBEDDING_MODEL_KEY: embedding_model},
        cluster_embedding_model=cfg_tree.CLUSTER_EMBEDDING_MODEL,
    )

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


def ingest_file(md_path: Path, out_dir: Path, force: bool, cfg_tree, cfg_sum) -> bool:
    """
    Index một file Markdown thành Tree và lưu file .pkl.

    Args:
        md_path: Đường dẫn tới file .md nguồn.
        out_dir: Thư mục lưu file .pkl đầu ra.
        force:   Ghi đè nếu file .pkl đã tồn tại.
        cfg_tree: TreeConfig instance.
        cfg_sum:  SummarizationConfig instance.

    Returns:
        True nếu thành công, False nếu bỏ qua hoặc lỗi.
    """
    pkl_path = out_dir / (md_path.stem + ".pkl")

    if pkl_path.exists() and not force:
        logger.info(f"[SKIP] Index đã tồn tại: {pkl_path} (dùng --force để ghi đè)")
        return False

    logger.info(f"[START] Đang index: {md_path.name}")

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            logger.warning(f"[WARN] File rỗng, bỏ qua: {md_path.name}")
            return False

        ra = _build_ra(cfg_tree, cfg_sum)

        # Bỏ qua warning "Overwriting existing tree" — đây là RA mới hoàn toàn
        ra.tree = None
        ra.tree = ra.tree_builder.build_from_text(markdown_text=text)

        from tree.retriever.tree_retriever import TreeRetriever
        ra.retriever = TreeRetriever(ra.tree_retriever_config, ra.tree)

        ra.save(str(pkl_path))
        logger.info(f"[DONE] Đã lưu index: {pkl_path}")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Lỗi khi index {md_path.name}: {e}", exc_info=True)
        return False


def main():
    from config import DATA_SRC_DIR, DATA_INDEX_DIR, TreeConfig, SummarizationConfig

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
        md_files = sorted(src_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"Không tìm thấy file .md nào trong: {src_dir}")
            sys.exit(0)

    logger.info(f"Tìm thấy {len(md_files)} file(s) cần index.")
    logger.info(f"  Nguồn : {src_dir}")
    logger.info(f"  Đầu ra: {out_dir}")
    logger.info(f"  Force : {args.force}")
    logger.info("-" * 60)

    # ── Index từng file ──────────────────────────────────────────────────────
    success, skipped, failed = 0, 0, 0
    for md_path in md_files:
        result = ingest_file(md_path, out_dir, args.force, cfg_tree, cfg_sum)
        if result is True:
            success += 1
        elif result is False and not args.force:
            # Phân biệt skip vs error qua log đã in trong ingest_file
            skipped += 1
        else:
            failed += 1

    logger.info("-" * 60)
    logger.info(
        f"Hoàn tất: {success} thành công | {skipped} bỏ qua | {failed} lỗi"
    )


if __name__ == "__main__":
    main()
