def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_db_without_database_url(client):
    resp = client.get("/api/v1/health/db")
    assert resp.status_code == 503
