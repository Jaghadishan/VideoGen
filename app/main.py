from fastapi import FastAPI

from app.api import chat, jobs, ws

app = FastAPI()

app.include_router(chat.router)
app.include_router(jobs.router)
app.include_router(ws.router)
