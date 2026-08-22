#!/usr/bin/env python3
"""Generate an image from a text prompt using a local diffusion model.

Self-contained general-purpose text-to-image generator. The model is
referenced by its Hugging Face repo ID and downloaded/cached on first use;
the script has no knowledge of any specific machine layout. Run it through
the memory-limiting wrapper ``gen_image.sh`` so it can never OOM-kill other
applications.

Usage:
  python generate_image.py --prompt "a cat" --out cat.png
  python generate_image.py --prompt "a cat" --size 1024 --steps 8 \\
      --seed 7 --out cat.png
"""

import argparse
import time

import torch
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

DEFAULT_NEGATIVE = "text, watermark, logo, cartoon, low quality, blurry"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an image with a local diffusion model."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text description of the image to generate",
    )
    parser.add_argument(
        "--negative", default=DEFAULT_NEGATIVE, help="Negative prompt"
    )
    parser.add_argument(
        "--out",
        default="/tmp/opencode/generated.png",
        help="Output image path",
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="Output image size (square)"
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="Number of inference steps"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID} (this may take a minute)...", flush=True)
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    # Text encoder + DiT cannot both live on a 16 GB card at once;
    # model_cpu_offload keeps only the active component in VRAM. Do NOT call
    # pipe.to("cuda") first - that moves everything at once and OOMs.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    print(f"Pipeline ready in {time.time() - t0:.0f}s", flush=True)

    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
    out.images[0].save(args.out)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
