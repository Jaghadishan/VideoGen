import logging
import shutil
import subprocess
from pathlib import Path

from app import config
from app.progress import events
from app.pipeline.video.base import VideoBackend
from app.pipeline.video.cogvideox import CogVideoX2B
from app.pipeline.video.ltx import LTX23
from app.pipeline.video.wan import Wan22TI2V5B, Wan2214B
from app.queue.models import Job, Shot

logger = logging.getLogger(__name__)

# Quality-first: for single/few-shot jobs, where the time cost is bounded.
# HunyuanVideo 1.5 (the spec's intended default) is currently EXCLUDED: its
# variable-length attention has no Windows flash/sage kernel and runs at
# ~200-1000 s/step on the 4070 (hours per clip). The backend is implemented
# (pipeline/video/hunyuan.py) — re-add it here once on flash-attn-capable
# hardware. So the effective default is Wan 2.2 TI2V-5B.
CHAIN = [Wan22TI2V5B(), CogVideoX2B(), LTX23()]

# Speed-first: for multi-shot jobs (see Multi-Shot Video Assembly in Specs.md).
MULTI_SHOT_CHAIN = [CogVideoX2B(), LTX23(), Wan22TI2V5B()]

# HunyuanVideo15 and Wan 2.2 14B are deliberately in neither automatic chain —
# Hunyuan for the speed reason above; Wan 14B is the explicit "maximum quality"
# trigger, used as a chain of one (no fallback) when job.max_quality is set.
MAX_QUALITY_CHAIN = [Wan2214B()]


def _select_chain(shots: list[Shot], max_quality: bool = False) -> list[VideoBackend]:
    if max_quality:
        return MAX_QUALITY_CHAIN
    return MULTI_SHOT_CHAIN if len(shots) >= config.MULTI_SHOT_THRESHOLD else CHAIN


def estimate_seconds(shots: list[Shot], max_quality: bool = False) -> float:
    chain = _select_chain(shots, max_quality=max_quality)
    per_clip = config.VIDEO_MODEL_ETA_SECONDS.get(chain[0].name, 0)
    return len(shots) * per_clip


def generate(job: Job, work_dir: Path) -> str:
    shots = job.brief.shots
    chain = _select_chain(shots, max_quality=job.max_quality)

    last_error: Exception | None = None
    for backend in chain:
        events.publish_step_change(job, sub_status=f"trying {backend.name}")
        try:
            _generate_all_shots(job, shots, backend, work_dir)
            return backend.name
        except Exception as exc:
            logger.warning("Video backend %s failed for job %s: %s", backend.name, job.job_id, exc)
            last_error = exc
        finally:
            # Free the GPU before the next backend attempt or the audio stage —
            # 12GB can't hold two models at once.
            backend.release_vram()

    raise RuntimeError(f"All video backends failed for job {job.job_id}") from last_error


def _generate_all_shots(job: Job, shots: list[Shot], backend: VideoBackend, work_dir: Path) -> None:
    clip_paths: list[Path] = []
    reference_image: Path | None = None

    for index, shot in enumerate(shots):
        clip_path = work_dir / f"shot_{index:03d}.mp4"
        backend.generate(job, shot, clip_path, reference_image=reference_image)
        clip_paths.append(clip_path)
        if index < len(shots) - 1:
            reference_image = _extract_last_frame(clip_path, work_dir / f"shot_{index:03d}_last_frame.png")

    _concatenate(clip_paths, work_dir / config.RAW_VIDEO_FILENAME)


def _extract_last_frame(video_path: Path, output_path: Path) -> Path:
    result = subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video_path), "-update", "1", "-q:v", "1", str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg last-frame extraction failed: {result.stderr}")
    return output_path


def _concatenate(clip_paths: list[Path], output_path: Path) -> None:
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], output_path)
        return

    list_file = output_path.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{path.name}'" for path in clip_paths))

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_path)],
        capture_output=True,
        text=True,
        cwd=list_file.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concatenation failed: {result.stderr}")
