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

    def release_vram(self) -> None:
        """Free GPU memory held between pipeline stages (video model must be off
        the GPU before an audio model loads — 12GB can't hold both). Backends
        that don't touch the GPU leave this as a no-op."""


def build_shot_prompt(job: Job, shot: Shot) -> str:
    """Compose a single shot's text prompt: the overarching, consistency-carrying
    visual description plus this shot's specific action, plus mood/style. Video
    models want one dense descriptive paragraph, not a list."""
    brief = job.brief
    parts = [brief.visual_description.strip(), shot.description.strip(), brief.mood_and_style.strip()]
    return " ".join(part for part in parts if part)
