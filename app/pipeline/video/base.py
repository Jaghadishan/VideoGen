from abc import ABC, abstractmethod
from pathlib import Path

from app.queue.models import Job, Shot


class VideoBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        """Write the generated clip to output_path.

        reference_image, if given, seeds image-to-video generation off the
        previous shot's last frame for continuity across a multi-shot job.
        """
