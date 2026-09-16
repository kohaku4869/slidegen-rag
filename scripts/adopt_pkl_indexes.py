"""
scripts/adopt_pkl_indexes.py — "Nhận nuôi" các file `.pkl` index cũ (build
bằng CLI, `data/index/*.pkl`) vào catalog Postgres mới (`documents`/
`document_indexes`) để chúng xuất hiện trong UI mà KHÔNG cần build lại từ
đầu (xem plan6.md Verification #15).

Bỏ qua `*_raptor.pkl` (backend raptor_tree đã gỡ, xem `main.py::select_index_file`)
và mọi tree `schema_version < 2` (thiếu `source_sha1`/heading metadata —
chạy `scripts/migrate_index_v2.py` trước).

Usage:
    uv run python scripts/adopt_pkl_indexes.py --dry-run   # chỉ in ra, không ghi DB
    uv run python scripts/adopt_pkl_indexes.py             # ghi thật
    uv run python scripts/adopt_pkl_indexes.py --verify    # đối chiếu .pkl <-> DB sau khi adopt
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import pickle
import sys
from pathlib import Path
from uuid import UUID

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from slidegen_rag.config import DATA_INDEX_DIR
from slidegen_rag.memory.catalog import PostgresDocumentRepository
from slidegen_rag.memory.engine import get_sessionmaker
from slidegen_rag.tree import legacy_pickle  # noqa: F401 — cài shim unpickle
from slidegen_rag.tree.stats import BuildStats

# users("Local")/projects("Inbox") — backfill của migration 0002.
INBOX_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000002")


def _discover_pkl_files(index_dir: Path) -> list[Path]:
    return sorted(f for f in index_dir.glob("*.pkl") if not f.stem.endswith("_raptor"))


def _extract_doc_summary_and_outline(tree) -> tuple[str, list[str]]:
    root = next(iter(tree.root_nodes.values()), None)
    doc_summary = root.text if root else ""
    outline_nodes = sorted(
        (n for n in tree.all_nodes.values() if n.depth == 0 and n.title),
        key=lambda n: n.line_start,
    )
    return doc_summary, [n.title for n in outline_nodes]


async def _adopt(index_dir: Path, *, dry_run: bool) -> None:
    document_repo = PostgresDocumentRepository(get_sessionmaker())
    files = _discover_pkl_files(index_dir)
    print(f"Tìm thấy {len(files)} file .pkl hợp lệ trong {index_dir} (đã loại *_raptor.pkl)")

    adopted, skipped = 0, 0
    for path in files:
        with open(path, "rb") as f:
            tree = pickle.load(f)

        schema_version = getattr(tree, "schema_version", 1)
        if schema_version < 2:
            print(f"[SKIP] {path.name}: schema_version={schema_version} < 2 — chạy scripts/migrate_index_v2.py trước")
            skipped += 1
            continue

        source_sha1 = getattr(tree, "source_sha1", "") or ""
        if not source_sha1:
            print(f"[SKIP] {path.name}: thiếu source_sha1")
            skipped += 1
            continue

        existing = await document_repo.find_by_sha1(
            project_id=INBOX_PROJECT_ID, source_sha1=source_sha1
        )
        if existing is not None:
            print(f"[SKIP] {path.name}: đã adopt trước đó (document {existing.id}, D{existing.ref_no})")
            skipped += 1
            continue

        doc_id = tree.doc_id or path.stem
        print(f"[ADOPT] {path.name} -> doc_id={doc_id}, sha1={source_sha1[:8]}…, nodes={len(tree.all_nodes)}")
        adopted += 1
        if dry_run:
            continue

        document = await document_repo.create_document(
            project_id=INBOX_PROJECT_ID,
            doc_slug=doc_id,
            title=doc_id,
            source_sha1=source_sha1,
            source_path=getattr(tree, "source_path", "") or str(path),
            source_bytes=path.stat().st_size,
        )
        index_row = await document_repo.start_index(document_id=document.id, source_sha1=source_sha1)

        doc_summary, outline = _extract_doc_summary_and_outline(tree)
        raw_stats = getattr(tree, "build_stats", None)
        build_stats = raw_stats.to_dict() if isinstance(raw_stats, BuildStats) else {}
        artifact_sha1 = hashlib.sha1(path.read_bytes()).hexdigest()

        await document_repo.finish_index(
            index_row.id,
            artifact_path=str(path),
            artifact_sha1=artifact_sha1,
            build_stats=build_stats,
            doc_summary=doc_summary,
            outline=outline,
            total_nodes=len(tree.all_nodes),
            num_layers=tree.num_layers,
            total_tokens=sum(n.n_tokens for n in tree.all_nodes.values()),
        )
        print(f"  -> document {document.id} (D{document.ref_no}) ready")

    verb = "sẽ adopt" if dry_run else "đã adopt"
    print(f"\nTổng kết: {adopted} {verb}, {skipped} bỏ qua.")


async def _verify(index_dir: Path) -> None:
    document_repo = PostgresDocumentRepository(get_sessionmaker())
    files = _discover_pkl_files(index_dir)
    documents = await document_repo.list_documents(project_id=INBOX_PROJECT_ID)
    adopted_slugs = {d.doc_slug for d in documents}

    ok, missing, skipped = 0, 0, 0
    for path in files:
        with open(path, "rb") as f:
            tree = pickle.load(f)
        if getattr(tree, "schema_version", 1) < 2:
            skipped += 1
            continue
        doc_id = tree.doc_id or path.stem
        if doc_id in adopted_slugs:
            print(f"[OK] {path.name} (doc_id={doc_id})")
            ok += 1
        else:
            print(f"[MISSING] {path.name} (doc_id={doc_id}) — chưa có trong catalog")
            missing += 1
    print(f"\nTổng kết: {ok} OK, {missing} missing, {skipped} bỏ qua (schema_version < 2).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir", default=None, help=f"Mặc định: {DATA_INDEX_DIR}"
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không ghi DB.")
    parser.add_argument(
        "--verify", action="store_true", help="Đối chiếu .pkl trên đĩa với catalog, không ghi DB."
    )
    args = parser.parse_args()
    index_dir = Path(args.index_dir) if args.index_dir else DATA_INDEX_DIR

    if args.verify:
        asyncio.run(_verify(index_dir))
    else:
        asyncio.run(_adopt(index_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
