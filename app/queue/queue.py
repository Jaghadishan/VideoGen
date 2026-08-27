from datetime import datetime

from app.progress import events
from app.queue import store
from app.queue.models import Brief, Job, JobStatus

ACTIVE_STATUSES = {
    JobStatus.WRITING_SCRIPT,
    JobStatus.GENERATING_VIDEO,
    JobStatus.GENERATING_AUDIO,
    JobStatus.MUXING,
}


def submit(brief: Brief) -> Job:
    job = Job(job_id=store.next_job_id(), created_at=datetime.now(), brief=brief)
    store.save_job(job)
    events.refresh_queue_positions(pending_jobs())
    return job


def pending_jobs() -> list[Job]:
    return [job for job in store.list_jobs() if job.status == JobStatus.PENDING]


def active_job() -> Job | None:
    for job in store.list_jobs():
        if job.status in ACTIVE_STATUSES:
            return job
    return None


def next_pending() -> Job | None:
    pending = pending_jobs()
    return pending[0] if pending else None


def queue_position(job_id: str) -> int | None:
    for index, job in enumerate(pending_jobs(), start=1):
        if job.job_id == job_id:
            return index
    return None
