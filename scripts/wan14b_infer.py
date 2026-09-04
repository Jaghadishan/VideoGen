"""Standalone Wan 2.2 A14B (T2V, GGUF Q4) inference, run by the .venv-wan
interpreter as a subprocess (see app/pipeline/video/wan.py :: Wan2214B).

Must not import from `app`.

Wan 2.2 A14B is a Mixture-of-Experts: a high-noise transformer and a low-noise
transformer (~14B each), switched at `boundary_ratio` during denoising. Both are
loaded from Q4_K_M GGUF and group-offloaded; the UMT5 text encoder is reused
from the TI2V-5B download and also offloaded. Even so this is the slow
"maximum quality, don't mind waiting" path — not in the automatic chain.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusers-dir", required=True, help="Wan2.2-T2V-A14B-Diffusers dir (vae/scheduler/configs)")
    parser.add_argument("--high-gguf", required=True)
    parser.add_argument("--low-gguf", required=True)
    parser.add_argument("--text-encoder-dir", required=True, help="dir with text_encoder/ + tokenizer/ (the TI2V-5B download)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--guidance-2", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from diffusers import (
        AutoencoderKLWan,
        GGUFQuantizationConfig,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import export_to_video
    from transformers import AutoTokenizer, UMT5EncoderModel

    quant = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    def load_expert(gguf_path: str, subfolder: str) -> WanTransformer3DModel:
        return WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=quant,
            config=args.diffusers_dir,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        )

    transformer = load_expert(args.high_gguf, "transformer")
    transformer_2 = load_expert(args.low_gguf, "transformer_2")

    vae = AutoencoderKLWan.from_pretrained(args.diffusers_dir, subfolder="vae", torch_dtype=torch.float32)
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.text_encoder_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_dir, subfolder="tokenizer")

    pipe = WanPipeline.from_pretrained(
        args.diffusers_dir,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)

    # Component-level CPU offload: only one module (a single 14B expert, or the
    # text encoder, or the VAE) is on the GPU at a time. The MoE uses just one
    # expert per timestep range, so this keeps peak VRAM ~ one Q4 expert.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        guidance_scale_2=args.guidance_2,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(frames, args.output, fps=args.fps)
    print(f"wan14b_infer: wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
