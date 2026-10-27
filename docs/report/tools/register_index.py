"""Gắn data/index/pldc_v2.pkl (build sẵn) vào document demo, dùng đúng hàm finalize của app."""
import asyncio, hashlib, shutil
from collections import OrderedDict
from pathlib import Path
from uuid import UUID
from sqlalchemy import text
from slidegen_rag.memory.engine import get_sessionmaker
from slidegen_rag.memory.catalog import PostgresDocumentRepository
from slidegen_rag.services.ingest_service import index_out_dir, _extract_doc_summary_and_outline
from slidegen_rag.services.tree_cache import load_tree

PID = UUID("01a17403-6bbe-7d4d-8e35-734710be129f")

async def main():
    sm = get_sessionmaker()
    async with sm() as db:
        doc = (await db.execute(text("select id from documents where project_id=:p and deleted_at is null"), {"p": PID})).one()
        idx = (await db.execute(text("select id,status from document_indexes where document_id=:d order by attempt_no desc limit 1"), {"d": doc.id})).one()
    print("doc", doc.id, "index", idx.id, idx.status)
    out = index_out_dir(PID); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{doc.id}.pkl"
    shutil.copy("data/index/pldc_v2.pkl", dst)
    tree = load_tree(OrderedDict(), str(dst))
    summ, outline = _extract_doc_summary_and_outline(tree)
    await PostgresDocumentRepository(sm).finish_index(
        idx.id, artifact_path=str(dst), artifact_sha1=hashlib.sha1(dst.read_bytes()).hexdigest(),
        build_stats={"note": "index build sẵn từ data/index/pldc_v2.pkl (CLI slidegen-ingest)"},
        doc_summary=summ, outline=outline, total_nodes=len(tree.all_nodes), num_layers=tree.num_layers,
        total_tokens=sum(n.n_tokens for n in tree.all_nodes.values()))
    print("nodes", len(tree.all_nodes), "layers", tree.num_layers)

asyncio.run(main())
