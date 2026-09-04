import logging
import os
import subprocess
from pathlib import Path

from app import config
from app.pipeline.video.base import VideoBackend, build_shot_prompt
from app.queue.models import Job, Shot

logger = logging.getLogger(__name__)


class HunyuanVideo15(VideoBackend):
    """HunyuanVideo 1.5 (480p, text-to-video). The quality-first default for
    single-shot jobs — strongest face/motion rendering, but heavy: runs
    scripts/hunyuan_infer.py under .venv-wan (diffusers >=0.36) with fp8 +
    4-bit + offload. Text-to-video only; multi-shot jobs skip this backend
    entirely (see pipeline/video/fallback.py), so continuity I2V isn't needed
    here."""

    name = "hunyuan_1.5"

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        model_dir = config.HUNYUAN_T2V_PATH
        wan_python = config.WAN_PYTHON  # shared with the Wan backend
        script = config.HUNYUAN_INFER_SCRIPT
        if not model_dir.exists() or not wan_python.exists() or not script.exists():
            raise FileNotFoundError(
                f"HunyuanVideo 1.5 not set up — need weights at {model_dir}, the "
                f".venv-wan python at {wan_python}, and {script}. See 4070-setup.md."
            )

        if reference_image is not None:
            logger.warning(
                "HunyuanVideo 1.5 backend is text-to-video only — ignoring reference image for job %s",
                job.job_id,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_shot_prompt(job, shot)

        cmd = [
            str(wan_python.resolve()),
            str(script.resolve()),
            "--model-dir", str(model_dir.resolve()),
            "--prompt", prompt,
            "--negative-prompt", config.HUNYUAN_NEGATIVE_PROMPT,
            "--output", str(output_path.resolve()),
            "--frames", str(config.HUNYUAN_NUM_FRAMES),
            "--steps", str(config.HUNYUAN_NUM_INFERENCE_STEPS),
            "--fps", str(config.HUNYUAN_FPS),
            "--attention-backend", config.HUNYUAN_ATTENTION_BACKEND,
        ]

        logger.info("HunyuanVideo 1.5 generating shot for job %s: %s", job.job_id, prompt[:120])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.HUNYUAN_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise RuntimeError(f"HunyuanVideo 1.5 inference failed (exit {result.returncode}):\n{tail}")
        if not output_path.exists():
            raise RuntimeError(f"HunyuanVideo 1.5 reported success but wrote no file at {output_path}")
