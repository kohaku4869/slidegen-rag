def test_job_not_found_404(client):
    resp = client.get("/api/v1/jobs/does-not-exist")
    assert resp.status_code == 404


def test_job_queued_snapshot(client):
    project_id = client.post("/api/v1/projects", json={"name": "P"}).json()["id"]
    document_id = client.post(
        f"/api/v1/projects/{project_id}/documents",
        data={"title": "Doc"},
        files={"file": ("doc.md", b"# H", "text/markdown")},
    ).json()["id"]
    job_id = client.post(f"/api/v1/documents/{document_id}/index").json()["job_id"]

    resp = client.get(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "queued"
    assert body["finalized"] is False


def test_job_failed_finalizes_document_index(client):
    project_id = client.post("/api/v1/projects", json={"name": "P"}).json()["id"]
    document_id = client.post(
        f"/api/v1/projects/{project_id}/documents",
        data={"title": "Doc"},
        files={"file": ("doc.md", b"# H", "text/markdown")},
    ).json()["id"]
    job_id = client.post(f"/api/v1/documents/{document_id}/index").json()["job_id"]

    job = client.app.state.job_registry.get(job_id)
    job.state = "failed"
    job.error = "boom"

    resp = client.get(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert body["error"] == "boom"
    assert body["finalized"] is True

    detail = client.get(f"/api/v1/documents/{document_id}").json()
    assert detail["document"]["status"] == "failed"
