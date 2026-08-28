from pathlib import Path

from app.pipeline.audio.base import AudioBackend
from app.queue.models import Job


class DiffRhythmPlus(AudioBackend):
    name = "diffrhythm_plus"

    def generate(self, job: Job, work_dir: Path) -> None:
        raise NotImplementedError("DiffRhythm+ backend not yet implemented")


def generate(job: Job, work_dir: Path) -> str:
    backend = DiffRhythmPlus()
    backend.generate(job, work_dir)
    return backend.name
