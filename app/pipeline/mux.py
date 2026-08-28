import shutil
import subprocess
from pathlib import Path

from app import config
from app.queue.models import Job


def combine(job: Job, work_dir: Path, output_path: Path) -> None:
    video_path = work_dir / config.RAW_VIDEO_FILENAME
    audio_path = work_dir / config.RAW_AUDIO_FILENAME

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_path.exists():
        _mux_with_audio(video_path, audio_path, output_path)
    else:
        shutil.copyfile(video_path, output_path)


def _mux_with_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg muxing failed: {result.stderr}")
