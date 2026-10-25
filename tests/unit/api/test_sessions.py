def _create_project(client, name="Demo"):
    return client.post("/api/v1/projects", json={"name": name}).json()["id"]


def test_create_and_get_session(client):
    project_id = _create_project(client)
    resp = client.post(
        f"/api/v1/projects/{project_id}/sessions", json={"title": "Chat 1"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "Chat 1"
    assert body["scope_document_ids"] == []

    resp = client.get(f"/api/v1/sessions/{body['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == body["id"]


def test_get_missing_session_404(client):
    resp = client.get("/api/v1/sessions/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


def test_list_sessions_for_project(client):
    project_id = _create_project(client)
    client.post(f"/api/v1/projects/{project_id}/sessions", json={"title": "A"})
    client.post(f"/api/v1/projects/{project_id}/sessions", json={"title": "B"})
    resp = client.get(f"/api/v1/projects/{project_id}/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_rename_archive_delete_session(client):
    project_id = _create_project(client)
    session_id = client.post(f"/api/v1/projects/{project_id}/sessions", json={}).json()[
        "id"
    ]

    resp = client.patch(f"/api/v1/sessions/{session_id}", json={"title": "Renamed"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Renamed"

    resp = client.post(f"/api/v1/sessions/{session_id}/archive")
    assert resp.status_code == 204

    resp = client.delete(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 204
    resp = client.get(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 404


def test_set_scope_and_list_messages_empty(client):
    project_id = _create_project(client)
    document_id = client.post(
        f"/api/v1/projects/{project_id}/documents",
        data={"title": "Doc"},
        files={"file": ("doc.md", b"# H", "text/markdown")},
    ).json()["id"]
    session_id = client.post(f"/api/v1/projects/{project_id}/sessions", json={}).json()[
        "id"
    ]

    resp = client.put(
        f"/api/v1/sessions/{session_id}/scope", json={"document_ids": [document_id]}
    )
    assert resp.status_code == 204

    resp = client.get(f"/api/v1/sessions/{session_id}")
    assert resp.json()["scope_document_ids"] == [document_id]

    resp = client.get(f"/api/v1/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == []
