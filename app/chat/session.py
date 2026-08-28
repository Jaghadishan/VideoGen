import threading
import uuid

from pydantic import BaseModel, Field

from app.chat import prompts


class Message(BaseModel):
    role: str
    content: str


class Session(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def get_or_create(session_id: str | None) -> str:
    with _lock:
        if session_id and session_id in _sessions:
            return session_id

        new_id = session_id or str(uuid.uuid4())
        opening = Message(role="assistant", content=prompts.OPENING_MESSAGE)
        _sessions[new_id] = Session(session_id=new_id, messages=[opening])
        return new_id


def add_message(session_id: str, role: str, content: str) -> None:
    with _lock:
        _sessions[session_id].messages.append(Message(role=role, content=content))


def history(session_id: str) -> list[Message]:
    with _lock:
        return list(_sessions[session_id].messages)


def transcript(session_id: str) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in history(session_id))
