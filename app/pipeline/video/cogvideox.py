import gc
import logging
import threading
from pathlib import Path

from app import config
from app.pipeline.video.base import VideoBackend, build_shot_prompt
from app.queue.models import Job, Shot

logger = logging.getLogger(__name__)


class CogVideoX2B(VideoBackend):
    """CogVideoX-2B text-to-video (diffusers). Lightest/fastest backend in the
    fallback chain. The 2B checkpoint is text-to-video only — there is no 2B
    image-to-video model — so it ignores the continuity reference_image and logs
    a warning; multi-shot continuity relies on the other backends for now."""

    name = "cogvideox_2b"

    _pipe = None
    _lock = threading.Lock()

    def _get_pipe(self):
        import torch
        from diffusers import CogVideoXPipeline

        with self._lock:
            if CogVideoX2B._pipe is None:
                model_path = config.COGVIDEOX_2B_PATH
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"CogVideoX-2B weights not found at {model_path} — "
                        f"run: hf download THUDM/CogVideoX-2b --local-dir {model_path}"
                    )
                logger.info("Loading CogVideoX-2B from %s", model_path)
                pipe = CogVideoXPipeline.from_pretrained(
                    str(model_path), torch_dtype=torch.float16, local_files_only=True
                )
                # Fit 12GB: offload modules to CPU until needed, tile/slice the VAE
                # decode. ~11GB peak (see the diffusers memory table).
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
                CogVideoX2B._pipe = pipe
            return CogVideoX2B._pipe

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        import torch
        from diffusers.utils import export_to_video

        if reference_image is not None:
            logger.warning(
                "CogVideoX-2B is text-to-video only — ignoring continuity reference image for job %s",
                job.job_id,
            )

        pipe = self._get_pipe()
        prompt = build_shot_prompt(job, shot)
        logger.info("CogVideoX-2B generating shot for job %s: %s", job.job_id, prompt[:120])

        generator = torch.Generator(device="cpu").manual_seed(0)
        result = pipe(
            prompt=prompt,
            negative_prompt=config.COGVIDEOX_NEGATIVE_PROMPT,
            num_frames=config.COGVIDEOX_NUM_FRAMES,
            num_inference_steps=config.COGVIDEOX_NUM_INFERENCE_STEPS,
            guidance_scale=config.COGVIDEOX_GUIDANCE_SCALE,
            generator=generator,
        )
        frames = result.frames[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=config.COGVIDEOX_FPS)

    def release_vram(self) -> None:
        pipe = CogVideoX2B._pipe
        if pipe is None:
            return
        try:
            import torch

            # Pull offloaded submodules back off the GPU; keep the pipe in CPU RAM
            # so the next job doesn't pay the reload cost.
            if hasattr(pipe, "maybe_free_model_hooks"):
                pipe.maybe_free_model_hooks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.exception("CogVideoX-2B VRAM release failed")
