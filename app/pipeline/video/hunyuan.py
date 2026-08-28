from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job


class HunyuanVideo15(VideoBackend):
    name = "hunyuan_1.5"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("HunyuanVideo 1.5 backend not yet implemented")
