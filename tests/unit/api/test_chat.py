def _create_project_and_session(client):
    project_id = client.post("/api/v1/projects", json={"name": "P"}).json()["id"]
    session_id = client.post(f"/api/v1/projects/{project_id}/sessions", json={}).json()[
        "id"
    ]
    return project_id, session_id


def test_chat_without_ready_documents_returns_409(client):
    _project_id, session_id = _create_project_and_session(client)
    resp = client.post(
        f"/api/v1/sessions/{session_id}/chat", json={"query": "Xin chào"}
    )
    assert resp.status_code == 409


def test_chat_missing_session_404(client):
    resp = client.post(
        "/api/v1/sessions/00000000-0000-0000-0000-000000000099/chat",
        json={"query": "Xin chào"},
    )
    assert resp.status_code == 404


def test_chat_happy_path(client, monkeypatch):
    """Bỏ qua LangGraph/LLM thật — stub `get_merged_ra`/`run_agent_turn` ở
    tầng service, chỉ kiểm tra orchestration (contextualize/record/response)."""
    from slidegen_rag.api.services import chat_service

    class _FakeTree:
        def __init__(self):
            self.all_nodes: dict = {}

    class _FakeRa:
        tree = _FakeTree()

    async def _fake_get_merged_ra(*args, **kwargs):
        return _FakeRa()

    async def _fake_run_agent_turn(*args, **kwargs):
        return {
            "generation": "Đây là câu trả lời test.",
            "documents": [],
            "document_node_ids": [],
            "document_priorities": [],
        }

    monkeypatch.setattr(chat_service, "get_merged_ra", _fake_get_merged_ra)
    monkeypatch.setattr(chat_service, "run_agent_turn", _fake_run_agent_turn)

    _project_id, session_id = _create_project_and_session(client)
    resp = client.post(
        f"/api/v1/sessions/{session_id}/chat", json={"query": "Xin chào"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Đây là câu trả lời test."
    assert body["standalone_query"] == "Xin chào"
    assert body["citations"] == []

    messages = client.get(f"/api/v1/sessions/{session_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Xin chào"
    assert messages[1]["content"] == "Đây là câu trả lời test."
