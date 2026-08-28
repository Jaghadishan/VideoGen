from abc import ABC, abstractmethod
from pathlib import Path

from app.queue.models import Job


class VideoBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, job: Job, work_dir: Path) -> None:
        """Write the generated clip to work_dir / config.RAW_VIDEO_FILENAME."""
