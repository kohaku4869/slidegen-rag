"""Unit tests cho `InMemoryProjectRepository`/`InMemoryDocumentRepository`
(catalog_store.py) — implementation thật thứ hai của
`ProjectRepository`/`DocumentRepository` (catalog.py), không cần Postgres.
Xem plan6.md B2."""

from uuid import uuid4

from slidegen_rag.memory.catalog import DocumentRepository, ProjectRepository
from slidegen_rag.memory.catalog_store import (
    InMemoryDocumentRepository,
    InMemoryProjectRepository,
)
from slidegen_rag.memory.memory_store import InMemorySessionRepository


def test_project_repo_conforms_to_protocol():
    assert isinstance(InMemoryProjectRepository(), ProjectRepository)


def test_document_repo_conforms_to_protocol():
    assert isinstance(InMemoryDocumentRepository(), DocumentRepository)


async def test_create_project_and_list_excludes_deleted():
    repo = InMemoryProjectRepository()
    owner = uuid4()
    p1 = await repo.create_project(owner_id=owner, name="Dự án A")
    p2 = await repo.create_project(owner_id=owner, name="Dự án B")
    await repo.soft_delete_project(p2.id)

    rows = await repo.list_projects(owner_id=owner)
    assert [r.id for r in rows] == [p1.id]

    rows_all = await repo.list_projects(owner_id=owner, include_deleted=True)
    assert {r.id for r in rows_all} == {p1.id, p2.id}


async def test_rename_project():
    repo = InMemoryProjectRepository()
    p = await repo.create_project(owner_id=uuid4(), name="Cũ")
    await repo.rename_project(p.id, name="Mới")
    got = await repo.get_project(p.id)
    assert got.name == "Mới"


async def test_create_document_assigns_sequential_ref_no():
    repo = InMemoryDocumentRepository()
    project_id = uuid4()
    d1 = await repo.create_document(
        project_id=project_id, doc_slug="pldc", title="PLDC", source_sha1="a" * 40
    )
    d2 = await repo.create_document(
        project_id=project_id, doc_slug="other", title="Other", source_sha1="b" * 40
    )
    assert (d1.ref_no, d2.ref_no) == (1, 2)


async def test_find_by_sha1_detects_duplicate_upload():
    repo = InMemoryDocumentRepository()
    project_id = uuid4()
    sha1 = "c" * 40
    created = await repo.create_document(
        project_id=project_id, doc_slug="pldc", title="PLDC", source_sha1=sha1
    )
    found = await repo.find_by_sha1(project_id=project_id, source_sha1=sha1)
    assert found.id == created.id
    assert await repo.find_by_sha1(project_id=project_id, source_sha1="d" * 40) is None


async def test_index_lifecycle_ready_then_reindex_supersedes_previous():
    repo = InMemoryDocumentRepository()
    project_id = uuid4()
    doc = await repo.create_document(
        project_id=project_id, doc_slug="pldc", title="PLDC", source_sha1="e" * 40
    )

    idx1 = await repo.start_index(document_id=doc.id, source_sha1="e" * 40)
    assert idx1.attempt_no == 1
    await repo.finish_index(
        idx1.id,
        artifact_path="data/index/pldc.pkl",
        artifact_sha1="f" * 40,
        build_stats={"nodes": 10},
        doc_summary="tóm tắt",
        outline=["1. Mở đầu"],
        total_nodes=10,
        num_layers=3,
        total_tokens=1000,
    )
    active = await repo.get_active_index(doc.id)
    assert active.id == idx1.id
    assert active.status == "ready"
    doc_after = await repo.get_document(doc.id)
    assert doc_after.status == "ready"
    assert doc_after.active_index_id == idx1.id

    idx2 = await repo.start_index(document_id=doc.id, source_sha1="e" * 40)
    assert idx2.attempt_no == 2
    await repo.finish_index(
        idx2.id,
        artifact_path="data/index/pldc.pkl",
        artifact_sha1="0" * 40,
        build_stats={"nodes": 12},
        doc_summary="tóm tắt 2",
        outline=["1. Mở đầu"],
        total_nodes=12,
        num_layers=3,
        total_tokens=1100,
    )
    idx1_after = await repo.get_active_index(doc.id)
    assert idx1_after.id == idx2.id

    doc_final = await repo.get_document(doc.id)
    assert doc_final.active_index_id == idx2.id
    assert doc_final.total_nodes == 12


async def test_fail_index_marks_document_failed_only_without_active_index():
    repo = InMemoryDocumentRepository()
    project_id = uuid4()
    doc = await repo.create_document(
        project_id=project_id, doc_slug="pldc", title="PLDC", source_sha1="a" * 40
    )
    idx1 = await repo.start_index(document_id=doc.id, source_sha1="a" * 40)
    await repo.fail_index(idx1.id, error_message="boom")

    doc_after = await repo.get_document(doc.id)
    assert doc_after.status == "failed"

    # Có bản ready rồi thì lần build lại hỏng KHÔNG được đổi status tài liệu.
    idx_ok = await repo.start_index(document_id=doc.id, source_sha1="a" * 40)
    await repo.finish_index(
        idx_ok.id,
        artifact_path="p",
        artifact_sha1="b" * 40,
        build_stats={},
        doc_summary="s",
        outline=[],
        total_nodes=1,
        num_layers=1,
        total_tokens=1,
    )
    idx3 = await repo.start_index(document_id=doc.id, source_sha1="a" * 40)
    await repo.fail_index(idx3.id, error_message="boom again")
    doc_final = await repo.get_document(doc.id)
    assert doc_final.status == "ready"
    assert doc_final.active_index_id == idx_ok.id


async def test_project_stats_aggregates_documents_and_sessions():
    document_repo = InMemoryDocumentRepository()
    session_repo = InMemorySessionRepository()
    project_repo = InMemoryProjectRepository(
        document_repo=document_repo, session_repo=session_repo
    )
    owner = uuid4()
    project = await project_repo.create_project(owner_id=owner, name="P")
    await document_repo.create_document(
        project_id=project.id, doc_slug="d1", title="D1", source_sha1="a" * 40
    )
    await document_repo.create_document(
        project_id=project.id, doc_slug="d2", title="D2", source_sha1="b" * 40
    )
    await session_repo.create_session(project_id=project.id)

    stats = await project_repo.project_stats(project.id)
    assert stats.document_count == 2
    assert stats.session_count == 1
