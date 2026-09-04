"""Standalone HunyuanVideo 1.5 (480p, text-to-video) inference, run by the
.venv-wan interpreter as a subprocess (see app/pipeline/video/hunyuan.py).

Must not import from `app` — a separate venv runs it.

HunyuanVideo 1.5 is 8.3B params with a Qwen2.5-VL-7B text encoder, so on a
12GB / 32GB box it needs: fp8 layerwise casting on the transformer, 4-bit
quantization of the text encoder, model CPU offload, and VAE tiling.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        default="",
        help="diffusers attention backend, e.g. flash_varlen_hub (fast, Linux) or "
        "_native_cudnn (best available on Windows). Empty = pipeline default.",
    )
    args = parser.parse_args()

    import torch
    from diffusers import AutoModel, HunyuanVideo15Pipeline, PipelineQuantizationConfig
    from diffusers.utils import export_to_video

    # 4-bit the Qwen2.5-VL text encoder (bf16 it is ~14GB resident; 4-bit ~5GB).
    quant = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["text_encoder"],
    )

    # fp8 layerwise weight-casting on the transformer (~16GB bf16 -> ~8GB stored).
    transformer = AutoModel.from_pretrained(args.model_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
    )

    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model_dir,
        transformer=transformer,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
    )
    if args.attention_backend:
        try:
            pipe.transformer.set_attention_backend(args.attention_backend)
        except Exception as exc:  # noqa: BLE001
            print(f"hunyuan_infer: could not set attention backend {args.attention_backend!r}: {exc}", file=sys.stderr)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]
    export_to_video(frames, args.output, fps=args.fps)
    print(f"hunyuan_infer: wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
