from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job


class CogVideoX2B(VideoBackend):
    name = "cogvideox_2b"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("CogVideoX-2B backend not yet implemented")
