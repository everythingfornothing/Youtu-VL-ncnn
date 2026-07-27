#!/usr/bin/env python3
"""Create frozen Youtu-VL Processor tensors for the C++ ncnn runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "tencent/Youtu-VL-4B-Instruct"
DEFAULT_REVISION = "8d30a0e49662a1d628a472b12df264dbcd768753"
IMAGE_TOKEN_ID = 128264


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_MODULES_CACHE", str(output_dir / ".hf-modules"))

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    names = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_attention_mask",
        "spatial_shapes",
    )
    arrays = {}
    for name in names:
        value = inputs[name].detach().cpu().numpy()
        value = value.astype(np.int64 if np.issubdtype(value.dtype, np.integer) else np.float32)
        value = np.ascontiguousarray(value)
        np.save(output_dir / f"{name}.npy", value, allow_pickle=False)
        arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    spatial = np.asarray(inputs["spatial_shapes"].cpu()).reshape(-1, 2)[0]
    grid = [int(spatial[0]), int(spatial[1])]
    if grid[0] % 2 or grid[1] % 2:
        raise RuntimeError(f"Vision grid must be divisible by merge size 2: {grid}")
    input_ids = np.load(output_dir / "input_ids.npy", allow_pickle=False)
    image_tokens = int(np.count_nonzero(input_ids == IMAGE_TOKEN_ID))
    expected_tokens = grid[0] * grid[1] // 4
    if image_tokens != expected_tokens:
        raise RuntimeError(f"image tokens {image_tokens} != expected {expected_tokens}")

    report = {
        "model": args.model,
        "revision": args.revision,
        "image": str(image),
        "prompt": args.prompt,
        "spatial_grid": grid,
        "image_token_id": IMAGE_TOKEN_ID,
        "image_token_count": image_tokens,
        "outputs": arrays,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
