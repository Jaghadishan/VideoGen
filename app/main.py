import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import chat, jobs, ws
from app.progress.events import bus
from app.queue import worker

INDEX_HTML = Path(__file__).parent / "web" / "templates" / "index.html"

app = FastAPI()

app.include_router(chat.router)
app.include_router(jobs.router)
app.include_router(ws.router)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.on_event("startup")
async def _startup() -> None:
    bus.bind_loop(asyncio.get_running_loop())
    threading.Thread(target=worker.run_forever, daemon=True).start()
