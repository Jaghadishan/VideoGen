import pytest
from fastapi.testclient import TestClient

from app.chat import session
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    session._sessions.clear()
    yield
    session._sessions.clear()


def test_get_messages_404_for_unknown_session():
    r = client.get("/chat/messages/does-not-exist")
    assert r.status_code == 404


def test_get_messages_returns_history_for_known_session():
    sid = session.get_or_create(None)
    session.add_message(sid, "user", "hi")

    body = client.get(f"/chat/messages/{sid}").json()

    assert [m["content"] for m in body][-1] == "hi"


def test_history_is_empty_not_error_for_unknown_session():
    assert session.history("gone") == []
    assert session.exists("gone") is False


def test_generate_404_when_session_expired():
    r = client.post("/jobs/generate", json={"session_id": "expired-after-restart"})
    assert r.status_code == 404
    assert "start a new conversation" in r.json()["detail"]


def test_generate_runs_when_session_exists(monkeypatch):
    from datetime import datetime

    from app.api import jobs
    from app.queue.models import AudioType, Brief, ContentPolicy, Shot

    sid = session.get_or_create(None)
    session.add_message(sid, "user", "a calm forest clip, ambient sound, no script")

    brief = Brief(
        title="Forest",
        visual_description="misty forest",
        shots=[Shot(description="trees")],
        audio_type=AudioType.AMBIENT,
        script_or_lyrics="",
        script_was_provided=True,
        mood_and_style="calm",
        target_length="short",
        content_policy=ContentPolicy.STANDARD,
    )
    monkeypatch.setattr("app.chat.llm.extract_brief", lambda session_id: (brief, False))

    body = client.post("/jobs/generate", json={"session_id": sid}).json()

    assert body["brief"]["title"] == "Forest"
    assert body["needs_script_draft"] is False
