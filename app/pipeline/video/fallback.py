import logging
from pathlib import Path

from app.pipeline.video.cogvideox import CogVideoX2B
from app.pipeline.video.hunyuan import HunyuanVideo15
from app.pipeline.video.ltx import LTX23
from app.pipeline.video.wan import Wan22TI2V5B
from app.queue.models import Job

logger = logging.getLogger(__name__)

# Default first, then the automatic fallback chain on failure (OOM, crash, timeout).
# Wan 2.2 14B is deliberately excluded — manual "maximum quality" trigger only.
CHAIN = [HunyuanVideo15(), Wan22TI2V5B(), CogVideoX2B(), LTX23()]


def generate(job: Job, work_dir: Path) -> str:
    last_error: Exception | None = None

    for backend in CHAIN:
        try:
            backend.generate(job, work_dir)
            return backend.name
        except Exception as exc:
            logger.warning("Video backend %s failed for job %s: %s", backend.name, job.job_id, exc)
            last_error = exc

    raise RuntimeError(f"All video backends failed for job {job.job_id}") from last_error
