from pathlib import Path

from app.pipeline.video.base import VideoBackend
from app.queue.models import Job, Shot


class LTX23(VideoBackend):
    name = "ltx_2.3"

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        raise NotImplementedError("LTX-2.3 backend not yet implemented")
