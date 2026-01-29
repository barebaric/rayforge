#!/usr/bin/env python3
"""ComfyUI client for image and 3D model generation."""

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path


COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
_default_output = (
    Path.home() / "Documents" / "projects" / "code" / "comfy" / "output"
)
COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", str(_default_output))


def create_text_to_image_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    model: str,
) -> dict:
    """Create a z-image-turbo text-to-image workflow."""
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    return {
        "1": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptySD3LatentImage",
        },
        "2": {
            "inputs": {
                "text": prompt,
                "clip": ["3", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "3": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
            },
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {
                "vae_name": "ae.safetensors",
            },
            "class_type": "VAELoader",
        },
        "5": {
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "6": {
            "inputs": {
                "model": ["5", 0],
                "shift": 3,
            },
            "class_type": "ModelSamplingAuraFlow",
        },
        "7": {
            "inputs": {
                "conditioning": ["2", 0],
            },
            "class_type": "ConditioningZeroOut",
        },
        "8": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["6", 0],
                "positive": ["2", 0],
                "negative": ["7", 0],
                "latent_image": ["1", 0],
            },
            "class_type": "KSampler",
        },
        "9": {
            "inputs": {
                "samples": ["8", 0],
                "vae": ["4", 0],
            },
            "class_type": "VAEDecode",
        },
        "10": {
            "inputs": {
                "filename_prefix": "z-image-turbo",
                "images": ["9", 0],
            },
            "class_type": "SaveImage",
        },
    }


def create_3d_from_image_workflow(
    image_path: str | Path,
    steps: int,
    cfg: float,
    seed: int,
    resolution: int,
) -> dict:
    """Create a Hunyuan3D image-to-3D workflow."""
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    image_path = Path(image_path)
    filename = image_path.name
    output_path = Path(COMFYUI_OUTPUT_DIR)
    comfy_input_dir = output_path.parent / "input"

    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dest_path = comfy_input_dir / filename

    if image_path.resolve() != dest_path.resolve():
        shutil.copy2(image_path, dest_path)

    return {
        "1": {
            "inputs": {"ckpt_name": "hunyuan_3d_v2.1.safetensors"},
            "class_type": "ImageOnlyCheckpointLoader",
        },
        "2": {
            "inputs": {"image": filename},
            "class_type": "LoadImage",
        },
        "3": {
            "inputs": {"model": ["1", 0], "shift": 1},
            "class_type": "ModelSamplingAuraFlow",
        },
        "4": {
            "inputs": {
                "clip_vision": ["1", 1],
                "image": ["2", 0],
                "crop": "center",
            },
            "class_type": "CLIPVisionEncode",
        },
        "5": {
            "inputs": {"clip_vision_output": ["4", 0]},
            "class_type": "Hunyuan3Dv2Conditioning",
        },
        "6": {
            "inputs": {"resolution": resolution, "batch_size": 1},
            "class_type": "EmptyLatentHunyuan3Dv2",
        },
        "7": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["6", 0],
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2],
                "num_chunks": 8000,
                "octree_resolution": 256,
            },
            "class_type": "VAEDecodeHunyuan3D",
        },
        "9": {
            "inputs": {
                "voxel": ["8", 0],
                "algorithm": "surface net",
                "threshold": 0.6,
            },
            "class_type": "VoxelToMesh",
        },
        "10": {
            "inputs": {"filename_prefix": "mesh/ComfyUI", "mesh": ["9", 0]},
            "class_type": "SaveGLB",
        },
    }


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = -1,
    model: str = "",
) -> str:
    """Generate an image using z-image-turbo text-to-image."""
    workflow = create_text_to_image_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        model=model,
    )

    prompt_data = json.dumps({"prompt": workflow}).encode("utf-8")
    url = f"{COMFYUI_URL}/prompt"
    req = urllib.request.Request(
        url, data=prompt_data, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    prompt_id = result["prompt_id"]
    return prompt_id


def generate_3d_from_image(
    image_path: str | Path,
    steps: int = 50,
    cfg: float = 7.0,
    seed: int = -1,
    resolution: int = 1024,
) -> str:
    """Generate a 3D model using Hunyuan3D image-to-3D."""
    workflow = create_3d_from_image_workflow(
        image_path=image_path,
        steps=steps,
        cfg=cfg,
        seed=seed,
        resolution=resolution,
    )

    prompt_data = json.dumps({"prompt": workflow}).encode("utf-8")
    url = f"{COMFYUI_URL}/prompt"
    req = urllib.request.Request(
        url, data=prompt_data, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    prompt_id = result["prompt_id"]
    return prompt_id


def get_history(prompt_id: str) -> dict:
    """Get execution history for a prompt."""
    url = f"{COMFYUI_URL}/history/{prompt_id}"

    with urllib.request.urlopen(url, timeout=300) as response:
        history = json.loads(response.read().decode("utf-8"))

    return history


def get_queue() -> dict:
    """Get current queue status."""
    url = f"{COMFYUI_URL}/queue"

    with urllib.request.urlopen(url, timeout=300) as response:
        queue_data = json.loads(response.read().decode("utf-8"))

    return queue_data


def clear_queue() -> None:
    """Clear current queue."""
    prompt_data = json.dumps({"clear": True}).encode("utf-8")
    url = f"{COMFYUI_URL}/queue"
    req = urllib.request.Request(
        url, data=prompt_data, headers={"Content-Type": "application/json"}
    )

    urllib.request.urlopen(req, timeout=300)


def get_models() -> list[str]:
    """List available checkpoint models."""
    try:
        url = f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple"

        with urllib.request.urlopen(url, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))

        if "CheckpointLoaderSimple" in data:
            required = data["CheckpointLoaderSimple"]["input"]["required"]
            if "ckpt_name" in required:
                return required["ckpt_name"][0]
    except Exception:
        pass

    return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ComfyUI client for image and 3D model generation"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    generate_image_parser = subparsers.add_parser(
        "generate-image",
        help="Generate an image using z-image-turbo text-to-image",
    )
    generate_image_parser.add_argument(
        "prompt", help="Text prompt for image generation"
    )
    generate_image_parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt for things to avoid",
    )
    generate_image_parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    generate_image_parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)",
    )
    generate_image_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of sampling steps (default: 4)",
    )
    generate_image_parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="Classifier free guidance scale (default: 1.0)",
    )
    generate_image_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random) (default: -1)",
    )
    generate_image_parser.add_argument(
        "--model",
        default="",
        help="Model checkpoint filename (empty for default)",
    )

    generate_3d_parser = subparsers.add_parser(
        "generate-3d", help="Generate a 3D model using Hunyuan3D image-to-3D"
    )
    generate_3d_parser.add_argument(
        "image_path", help="Path to image file for 3D model generation"
    )
    generate_3d_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps (default: 50)",
    )
    generate_3d_parser.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Classifier free guidance scale (default: 7.0)",
    )
    generate_3d_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random) (default: -1)",
    )
    generate_3d_parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution for 3D generation (1-8192) (default: 1024)",
    )

    get_history_parser = subparsers.add_parser(
        "get-history", help="Get execution history for a prompt"
    )
    get_history_parser.add_argument(
        "prompt_id", help="Prompt ID to get history for"
    )

    subparsers.add_parser("get-queue", help="Get current queue status")

    subparsers.add_parser("clear-queue", help="Clear current queue")

    subparsers.add_parser(
        "get-models", help="List available checkpoint models"
    )

    args = parser.parse_args()

    if args.command == "generate-image":
        prompt_id = generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            model=args.model,
        )
        print(
            f"z-image-turbo generation started. "
            f"Prompt ID: {prompt_id}\n"
            f"Use get-history tool to check status."
        )

    elif args.command == "generate-3d":
        prompt_id = generate_3d_from_image(
            image_path=args.image_path,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            resolution=args.resolution,
        )
        print(
            f"3D model generation from image started. "
            f"Prompt ID: {prompt_id}\n"
            f"Use get-history tool to check status."
        )

    elif args.command == "get-history":
        history = get_history(args.prompt_id)

        if args.prompt_id not in history:
            print(f"No history found for prompt ID: {args.prompt_id}")
            sys.exit(1)

        prompt_info = history[args.prompt_id]
        status = prompt_info.get("status", {})

        result_text = f"Prompt ID: {args.prompt_id}\n"
        result_text += f"Status: {json.dumps(status, indent=2)}\n"

        outputs = prompt_info.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    filename = img["filename"]
                    subfolder = img.get("subfolder", "")

                    if subfolder:
                        filepath = str(
                            Path(COMFYUI_OUTPUT_DIR) / subfolder / filename
                        )
                    else:
                        filepath = str(Path(COMFYUI_OUTPUT_DIR) / filename)

                    result_text += f"\nGenerated image: {filepath}\n"
            elif "mesh" in node_output:
                for mesh in node_output["mesh"]:
                    filename = mesh["filename"]
                    subfolder = mesh.get("subfolder", "")

                    if subfolder:
                        filepath = str(
                            Path(COMFYUI_OUTPUT_DIR) / subfolder / filename
                        )
                    else:
                        filepath = str(Path(COMFYUI_OUTPUT_DIR) / filename)

                    result_text += f"\nGenerated 3D model: {filepath}\n"

        print(result_text)

    elif args.command == "get-queue":
        queue_data = get_queue()

        queue_running = queue_data.get("queue_running", [])
        queue_pending = queue_data.get("queue_pending", [])

        result_text = f"Queue running: {len(queue_running)} items\n"
        result_text += f"Queue pending: {len(queue_pending)} items\n"

        if queue_running:
            result_text += "\nRunning:\n"
            for item in queue_running:
                prompt_id = item[1]
                result_text += f"  - Prompt ID: {prompt_id}\n"

        if queue_pending:
            result_text += "\nPending:\n"
            for item in queue_pending:
                prompt_id = item[1]
                result_text += f"  - Prompt ID: {prompt_id}\n"

        print(result_text)

    elif args.command == "clear-queue":
        clear_queue()
        print("Queue cleared successfully")

    elif args.command == "get-models":
        models = get_models()
        result_text = f"Available models ({len(models)}):\n"
        for model in models:
            result_text += f"  - {model}\n"
        print(result_text)


if __name__ == "__main__":
    main()
