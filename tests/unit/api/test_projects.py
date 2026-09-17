def test_create_and_get_project(client):
    resp = client.post("/api/v1/projects", json={"name": "Demo"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Demo"
    assert body["language"] == "vi"

    resp = client.get(f"/api/v1/projects/{body['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == body["id"]


def test_get_missing_project_404(client):
    resp = client.get("/api/v1/projects/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


def test_list_projects(client):
    client.post("/api/v1/projects", json={"name": "A"})
    client.post("/api/v1/projects", json={"name": "B"})
    resp = client.get("/api/v1/projects")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()}
    assert {"A", "B"} <= names


def test_rename_and_delete_project(client):
    created = client.post("/api/v1/projects", json={"name": "Old"}).json()
    project_id = created["id"]

    resp = client.patch(f"/api/v1/projects/{project_id}", json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"

    resp = client.delete(f"/api/v1/projects/{project_id}")
    assert resp.status_code == 204

    # Soft delete: get_project (như bản Postgres) không lọc deleted_at, chỉ
    # list_projects mới ẩn project đã xoá — xem memory/catalog.py.
    resp = client.get(f"/api/v1/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted_at"] is not None

    resp = client.get("/api/v1/projects")
    assert project_id not in {p["id"] for p in resp.json()}


def test_project_stats_empty(client):
    project_id = client.post("/api/v1/projects", json={"name": "Stats"}).json()["id"]
    resp = client.get(f"/api/v1/projects/{project_id}/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_count"] == 0
    assert body["session_count"] == 0
