"""Thin wrapper around fal.ai's flux-lora text-to-image endpoint.

We deliberately pin to a specific endpoint so generated assets are
reproducible across runs (same prompt + seed -> same image).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.request import urlretrieve

import fal_client

log = logging.getLogger(__name__)

DEFAULT_MODEL = "fal-ai/flux/dev"  # plain Flux dev, no LoRA dependency
DEFAULT_IMAGE_SIZE = "square_hd"   # 1024 x 1024
DEFAULT_INFERENCE_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 6.0  # higher than Flux default; we want strict prompt adherence


def ensure_key() -> str:
    key = os.environ.get("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FAL_KEY missing. Set it in .env or export it. "
            "Get a key at https://fal.ai/dashboard/keys"
        )
    return key


def generate_image(
    prompt: str,
    *,
    seed: int | None = None,
    model: str = DEFAULT_MODEL,
    image_size: str = DEFAULT_IMAGE_SIZE,
    steps: int = DEFAULT_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
) -> dict:
    """Submit a prompt and block until the image URL is ready."""
    ensure_key()
    args: dict = {
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images": 1,
        "enable_safety_checker": False,
    }
    if seed is not None:
        args["seed"] = int(seed)

    log.info("fal.ai submit  model=%s  seed=%s", model, seed)
    t0 = time.time()
    result = fal_client.subscribe(
        model,
        arguments=args,
        with_logs=False,
    )
    log.info("fal.ai done    elapsed=%.1fs", time.time() - t0)
    return result


def download_image(result: dict, dest: Path) -> Path:
    """Download the first image URL from a fal-client result into *dest*."""
    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"fal.ai response had no images: {result!r}")
    url = images[0].get("url")
    if not url:
        raise RuntimeError(f"fal.ai image entry missing url: {images[0]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    log.info("downloaded -> %s (%d KB)", dest.name, dest.stat().st_size // 1024)
    return dest
