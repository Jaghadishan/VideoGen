from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessageRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatMessageResponse(BaseModel):
    session_id: str
    reply: str


@router.post("/messages", response_model=ChatMessageResponse)
def send_message(request: ChatMessageRequest) -> ChatMessageResponse:
    from app.chat import llm, session

    session_id = session.get_or_create(request.session_id)
    session.add_message(session_id, "user", request.message)
    reply = llm.reply(session_id)
    session.add_message(session_id, "assistant", reply)
    return ChatMessageResponse(session_id=session_id, reply=reply)


@router.get("/messages/{session_id}")
def get_messages(session_id: str):
    from app.chat import session

    return session.history(session_id)
