import asyncio

from fastapi import FastAPI

from app.api import chat, jobs, ws
from app.progress.events import bus

app = FastAPI()

app.include_router(chat.router)
app.include_router(jobs.router)
app.include_router(ws.router)


@app.on_event("startup")
async def _bind_event_bus() -> None:
    bus.bind_loop(asyncio.get_running_loop())
