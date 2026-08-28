from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job, Shot


class CogVideoX2B(VideoBackend):
    name = "cogvideox_2b"

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        raise NotImplementedError("CogVideoX-2B backend not yet implemented")
