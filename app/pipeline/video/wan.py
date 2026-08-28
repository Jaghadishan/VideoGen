from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job


class Wan22TI2V5B(VideoBackend):
    name = "wan_2.2_ti2v_5b"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("Wan 2.2 TI2V-5B backend not yet implemented")


class Wan2214B(VideoBackend):
    """Manual 'maximum quality' trigger only — not part of the automatic fallback chain."""

    name = "wan_2.2_14b"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("Wan 2.2 14B backend not yet implemented")
