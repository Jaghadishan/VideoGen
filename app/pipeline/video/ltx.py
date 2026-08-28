from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job


class LTX23(VideoBackend):
    name = "ltx_2.3"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("LTX-2.3 backend not yet implemented")
