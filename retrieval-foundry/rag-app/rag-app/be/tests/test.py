import os

from fastapi.testclient import TestClient

from app.main import app

CONVERSATION_ID = "03783b51-8f61-4e6e-b13a-8fe5a7955b1e"

client = TestClient(
    app,
    headers={
        "X-API-Key": os.getenv("API_KEY", "")
    },
)


def test_reinvestment_period():
    response = client.post(
        "/api/v1/chat/ask",
        json={
            "question": "What is reinvestment period?",
            "file_ids": [
                "Redwood_CLO_2026-1",
                "BlueRiver_CLO_2026",
                "SilverBrook_CLO_2026_1",
            ],
            "created_by": {
                "analyst_name": "test-user",
                "email_id": "test@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_capital_structure():
    response = client.post(
        "/api/v1/chat/ask",
        json={
            "question": "What is the capital structure?",
            "file_ids": [
                "Summit_CLO_2026_2_document",
            ],
            "created_by": {
                "analyst_name": "test-user",
                "email_id": "test@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_reinvestment_period():
    response = client.post(
        "/api/v1/chat/ask",
        json={
            "question": "What is reinvestment period?",
            "file_ids": [
                "Redwood_CLO_2026-1",
                "BlueRiver_CLO_2026",
                "SilverBrook_CLO_2026_1",
            ],
            "created_by": {
                "analyst_name": "test-user",
                "email_id": "test@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_capital_structure():
    response = client.post(
        "/api/v1/chat/ask",
        json={
            "question": "What is the capital structure?",
            "file_ids": [
                "Summit_CLO_2026_2_document",
            ],
            "created_by": {
                "analyst_name": "test-user",
                "email_id": "test@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_list_conversations():
    response = client.get("/api/v1/conversations/")

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_get_conversation():
    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}"
    )

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_regenerate():
    response = client.post(
        "/api/v1/chat/regenerate",
        json={
            "conversation_id": CONVERSATION_ID,
            "question_id": "ff39f9b0-e603-4671-b839-ee7e2fb2709a",
            "edited_question": "What is the reinvestment period?",
            "created_by": {
                "analyst_name": "test-user",
                "email_id": "test@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] is True