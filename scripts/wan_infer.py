"""Standalone Wan 2.2 TI2V-5B inference, run by the .venv-wan interpreter as a
subprocess (see app/pipeline/video/wan.py).

This file must not import anything from `app` — a separate venv runs it, and that
venv only has torch + diffusers, not the project package.
"""

import argparse
import sys


def _build_pipeline(model_dir: str, want_image: bool):
    import inspect

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)

    pipe_cls = WanPipeline
    if want_image and "image" not in inspect.signature(WanPipeline.__call__).parameters:
        from diffusers import WanImageToVideoPipeline

        pipe_cls = WanImageToVideoPipeline

    pipe = pipe_cls.from_pretrained(model_dir, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001 - best effort, some builds lack it
        pass
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", default=None, help="reference image for image-to-video")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from diffusers.utils import export_to_video

    pipe = _build_pipeline(args.model_dir, want_image=bool(args.image))

    call_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.image:
        from diffusers.utils import load_image

        call_kwargs["image"] = load_image(args.image)

    frames = pipe(**call_kwargs).frames[0]
    export_to_video(frames, args.output, fps=args.fps)
    print(f"wan_infer: wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
