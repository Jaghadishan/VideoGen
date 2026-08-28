import subprocess
from pathlib import Path

from app import config
from app.pipeline.audio import song
from app.queue.models import Job


def _video_has_audio(video_path: Path) -> bool:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return output.strip() != ""


def generate(job: Job, work_dir: Path) -> str | None:
    video_path = work_dir / config.RAW_VIDEO_FILENAME
    if video_path.exists() and _video_has_audio(video_path):
        return None

    # Same model as the "song" stage, run in an instrumental-only mode once implemented.
    return song.generate(job, work_dir)
