import logging
import tempfile
import threading
import time
from pathlib import Path

from app import config
from app.queue import gpu, heartbeat, queue, store
from app.queue.models import AudioType, Job, JobStatus

POLL_INTERVAL_SECONDS = 2.0

logger = logging.getLogger(__name__)


def _transition(job: Job, status: JobStatus) -> None:
    job.status = status
    store.save_job(job)


def _run_stage(job: Job, status: JobStatus, step_name: str, fn) -> None:
    _transition(job, status)
    start = time.monotonic()
    fn()
    job.step_seconds[step_name] = time.monotonic() - start
    store.save_job(job)


def _write_script(job: Job) -> None:
    from app.chat import llm

    job.brief.script_or_lyrics = llm.write_script(job)


def _generate_video(job: Job, work_dir: Path) -> None:
    from app.pipeline.video import fallback

    job.video_model = fallback.generate(job, work_dir)


def _generate_audio(job: Job, work_dir: Path) -> None:
    audio_type = job.brief.audio_type
    if audio_type == AudioType.NONE:
        job.audio_model = None
        return

    if audio_type == AudioType.SONG:
        from app.pipeline.audio import song as backend
    elif audio_type == AudioType.VOICEOVER:
        from app.pipeline.audio import voiceover as backend
    else:
        from app.pipeline.audio import ambient as backend

    job.audio_model = backend.generate(job, work_dir)


def _mux(job: Job, work_dir: Path) -> None:
    from app.pipeline import mux

    output_path = store.job_dir(job.job_id) / config.VIDEO_OUTPUT_FILENAME
    mux.combine(job, work_dir, output_path)


def run_job(job: Job) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{job.job_id}_") as tmp:
        work_dir = Path(tmp)
        try:
            if job.needs_script_draft:
                _run_stage(job, JobStatus.WRITING_SCRIPT, "writing_script", lambda: _write_script(job))
            _run_stage(job, JobStatus.GENERATING_VIDEO, "generating_video", lambda: _generate_video(job, work_dir))
            _run_stage(job, JobStatus.GENERATING_AUDIO, "generating_audio", lambda: _generate_audio(job, work_dir))
            _run_stage(job, JobStatus.MUXING, "muxing", lambda: _mux(job, work_dir))
            _transition(job, JobStatus.DONE)
        except Exception as exc:
            job.error = str(exc)
            _transition(job, JobStatus.FAILED)
            logger.exception("Job %s failed", job.job_id)


def _recover_interrupted_jobs() -> None:
    for job in store.list_jobs():
        if job.status in queue.ACTIVE_STATUSES:
            job.error = "Worker restarted while this job was in progress."
            _transition(job, JobStatus.FAILED)


def run_forever(stop_event: threading.Event | None = None) -> None:
    stop_event = stop_event or threading.Event()
    _recover_interrupted_jobs()

    while not stop_event.is_set():
        heartbeat.beat()
        job = queue.next_pending() if gpu.gpu_available() else None
        if job is not None:
            run_job(job)
            continue
        stop_event.wait(POLL_INTERVAL_SECONDS)
