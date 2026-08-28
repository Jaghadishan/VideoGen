from abc import ABC, abstractmethod
from pathlib import Path

from app.queue.models import Job


class AudioBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, job: Job, work_dir: Path) -> None:
        """Write the generated audio to work_dir / config.RAW_AUDIO_FILENAME."""
