import asyncio
import threading

from fastapi import FastAPI

from app.api import chat, jobs, ws
from app.progress.events import bus
from app.queue import worker

app = FastAPI()

app.include_router(chat.router)
app.include_router(jobs.router)
app.include_router(ws.router)


@app.on_event("startup")
async def _startup() -> None:
    bus.bind_loop(asyncio.get_running_loop())
    threading.Thread(target=worker.run_forever, daemon=True).start()
