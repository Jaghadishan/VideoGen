from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config
from app.queue import queue, store
from app.queue.models import Brief, Job, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


class GenerateRequest(BaseModel):
    session_id: str


class BriefPreview(BaseModel):
    brief: Brief
    needs_script_draft: bool


class ConfirmRequest(BaseModel):
    brief: Brief
    needs_script_draft: bool = False


@router.post("/generate", response_model=BriefPreview)
def generate(request: GenerateRequest) -> BriefPreview:
    from app.chat import llm

    brief, needs_script_draft = llm.extract_brief(request.session_id)
    return BriefPreview(brief=brief, needs_script_draft=needs_script_draft)


@router.post("/confirm", response_model=Job)
def confirm(request: ConfirmRequest) -> Job:
    return queue.submit(request.brief, needs_script_draft=request.needs_script_draft)


@router.get("", response_model=list[Job])
def list_jobs() -> list[Job]:
    return store.list_jobs()


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    try:
        return store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job not found")


@router.get("/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail="job is not finished")

    path = store.job_dir(job_id) / config.VIDEO_OUTPUT_FILENAME
    return FileResponse(path, filename=f"{job_id}.mp4")
