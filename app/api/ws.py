from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.progress.events import bus

router = APIRouter(tags=["ws"])


@router.websocket("/ws/progress")
async def progress_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    subscriber = bus.subscribe()
    try:
        while True:
            event = await subscriber.get()
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(subscriber)
