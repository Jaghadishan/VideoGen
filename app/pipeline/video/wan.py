import logging
import os
import subprocess
from pathlib import Path

from app import config
from app.pipeline.video.base import VideoBackend, build_shot_prompt
from app.queue.models import Job, Shot

logger = logging.getLogger(__name__)


class Wan22TI2V5B(VideoBackend):
    """Wan 2.2 TI2V-5B (text/image-to-video). Needs a diffusers newer than the
    main env can hold, so it runs scripts/wan_infer.py under .venv-wan as a
    subprocess. Unlike CogVideoX-2B this model does real image-to-video, so it
    honours the multi-shot continuity reference frame."""

    name = "wan_2.2_ti2v_5b"

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        model_dir = config.WAN_TI2V_5B_PATH
        wan_python = config.WAN_PYTHON
        script = config.WAN_INFER_SCRIPT
        if not model_dir.exists() or not wan_python.exists() or not script.exists():
            raise FileNotFoundError(
                f"Wan 2.2 TI2V-5B not set up — need weights at {model_dir}, its venv "
                f"python at {wan_python}, and {script}. See 4070-setup.md."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_shot_prompt(job, shot)

        cmd = [
            str(wan_python.resolve()),
            str(script.resolve()),
            "--model-dir", str(model_dir.resolve()),
            "--prompt", prompt,
            "--negative-prompt", config.WAN_NEGATIVE_PROMPT,
            "--output", str(output_path.resolve()),
            "--frames", str(config.WAN_NUM_FRAMES),
            "--steps", str(config.WAN_NUM_INFERENCE_STEPS),
            "--guidance", str(config.WAN_GUIDANCE_SCALE),
            "--height", str(config.WAN_HEIGHT),
            "--width", str(config.WAN_WIDTH),
            "--fps", str(config.WAN_FPS),
        ]
        if reference_image is not None:
            cmd += ["--image", str(Path(reference_image).resolve())]

        logger.info(
            "Wan 2.2 TI2V-5B generating shot for job %s (%s): %s",
            job.job_id,
            "image-to-video" if reference_image else "text-to-video",
            prompt[:120],
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.WAN_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise RuntimeError(f"Wan 2.2 inference failed (exit {result.returncode}):\n{tail}")
        if not output_path.exists():
            raise RuntimeError(f"Wan 2.2 inference reported success but wrote no file at {output_path}")


class Wan2214B(VideoBackend):
    """Manual 'maximum quality' trigger only — not part of the automatic fallback chain."""

    name = "wan_2.2_14b"

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        raise NotImplementedError("Wan 2.2 14B backend not yet implemented")
