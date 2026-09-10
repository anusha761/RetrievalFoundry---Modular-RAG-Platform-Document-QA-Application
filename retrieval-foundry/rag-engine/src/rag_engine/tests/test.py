from fastapi.testclient import TestClient

from rag_engine.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_retrieve():
    response = client.post(
        "/retrieve",
        json={
            "user_query": "What is the reinvestment period?",
            "file_id": "Redwood_CLO_2026-1",
            "candidate_k": 4,
            "desired_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is the reinvestment period?"
    assert data["file_id"] == "Redwood_CLO_2026-1"
    assert data["candidate_k"] == 10
    assert data["desired_k"] == 5
    assert "results" in data


def test_retrieve_invalid_k():
    response = client.post(
        "/retrieve",
        json={
            "user_query": "What is the reinvestment period?",
            "file_id": "Redwood_CLO_2026-1",
            "candidate_k": 5,
            "desired_k": 10,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "desired_k cannot be greater than candidate_k."
    )


def test_chat():
    response = client.post(
        "/chat",
        json={
            "system_prompt": "You are a helpful financial assistant.",
            "user_prompt": "What is a CLO?",
            "conversation_history": [],
        },
    )

    assert response.status_code == 200
    assert "response" in response.json()
