import gc
import logging
import threading
from pathlib import Path

from app import config
from app.pipeline.video.base import VideoBackend, build_shot_prompt
from app.queue.models import Job, Shot

logger = logging.getLogger(__name__)


class LTX23(VideoBackend):
    """LTX-Video 2B (diffusers, in-process). Fast DiT with a high-compression
    Video-VAE. Supports real image-to-video, so it honours the multi-shot
    continuity reference frame. bf16 + model CPU offload + VAE tiling for 12GB."""

    name = "ltx_2.3"

    _pipe = None          # LTXPipeline (text-to-video)
    _i2v_pipe = None      # LTXImageToVideoPipeline, sharing the same modules
    _lock = threading.Lock()

    def _load(self):
        import torch
        from diffusers import LTXImageToVideoPipeline, LTXPipeline

        with self._lock:
            if LTX23._pipe is None:
                logger.info("Loading LTX-Video from %s", config.LTX_MODEL_ID)
                pipe = LTXPipeline.from_pretrained(config.LTX_MODEL_ID, torch_dtype=torch.bfloat16)
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()
                # I2V reuses the exact same module objects (and their offload hooks).
                LTX23._i2v_pipe = LTXImageToVideoPipeline(
                    vae=pipe.vae,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    transformer=pipe.transformer,
                    scheduler=pipe.scheduler,
                )
                LTX23._pipe = pipe
            return LTX23._pipe, LTX23._i2v_pipe

    def generate(self, job: Job, shot: Shot, output_path: Path, reference_image: Path | None = None) -> None:
        import torch
        from diffusers.utils import export_to_video

        t2v, i2v = self._load()
        prompt = build_shot_prompt(job, shot)
        logger.info(
            "LTX-Video generating shot for job %s (%s): %s",
            job.job_id,
            "image-to-video" if reference_image else "text-to-video",
            prompt[:120],
        )

        kwargs = dict(
            prompt=prompt,
            negative_prompt=config.LTX_NEGATIVE_PROMPT,
            height=config.LTX_HEIGHT,
            width=config.LTX_WIDTH,
            num_frames=config.LTX_NUM_FRAMES,
            frame_rate=config.LTX_FPS,
            num_inference_steps=config.LTX_NUM_INFERENCE_STEPS,
            guidance_scale=config.LTX_GUIDANCE_SCALE,
            decode_timestep=config.LTX_DECODE_TIMESTEP,
            decode_noise_scale=config.LTX_DECODE_NOISE_SCALE,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        if reference_image is not None:
            from diffusers.utils import load_image

            frames = i2v(image=load_image(str(reference_image)), **kwargs).frames[0]
        else:
            frames = t2v(**kwargs).frames[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=config.LTX_FPS)

    def release_vram(self) -> None:
        if LTX23._pipe is None:
            return
        try:
            import torch

            if hasattr(LTX23._pipe, "maybe_free_model_hooks"):
                LTX23._pipe.maybe_free_model_hooks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.exception("LTX-Video VRAM release failed")
