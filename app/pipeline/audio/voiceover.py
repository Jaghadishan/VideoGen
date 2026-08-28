from pathlib import Path

from app.pipeline.audio.base import AudioBackend
from app.queue.models import Job


class Kokoro82M(AudioBackend):
    name = "kokoro_82m"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("Kokoro-82M backend not yet implemented")


def generate(job: Job, work_dir: Path) -> str:
    backend = Kokoro82M()
    backend.generate(job, work_dir)
    return backend.name
