from fastapi.testclient import TestClient

from causalnews import api

client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_ok():
    response = client.post("/analyze", json={
        "text": "Vaccines definitely cause autism. Share this before it is deleted!"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "likely_fake"
    assert body["myth_matches"]


def test_analyze_rejects_short_text():
    assert client.post("/analyze", json={"text": "hi"}).status_code == 422


def test_analyze_rejects_huge_text():
    huge = "word " * (api.MAX_TEXT_LENGTH // 4)
    assert client.post("/analyze", json={"text": huge}).status_code == 422


def test_analyze_requires_text_field():
    assert client.post("/analyze", json={}).status_code == 422
