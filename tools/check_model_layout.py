#!/usr/bin/env python3
"""Validate the legacy model layout required by the current C++ runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_files(root: Path) -> list[Path]:
    files = [
        root / "artifacts/vision_embedding/vision_embedding_wrapper.ncnn.param",
        root / "artifacts/vision_embedding/vision_embedding_wrapper.ncnn.bin",
        root
        / "artifacts/vision_post_layernorm/vision_post_layernorm_wrapper.ncnn.param",
        root
        / "artifacts/vision_post_layernorm/vision_post_layernorm_wrapper.ncnn.bin",
        root / "artifacts/text_embedding/embed_tokens_weight.npy",
        root / "artifacts/llm_rope_inv_freq.npy",
        root / "models/youtu_merger.ncnn.param",
        root / "models/youtu_merger.ncnn.bin",
        root / "tokenizer/tokenizer.bin",
    ]
    for index in range(27):
        name = f"vision_layer{index}_masked_core_wrapper"
        directory = root / f"artifacts/vision_layer{index}_masked_core"
        files.extend((directory / f"{name}.ncnn.param", directory / f"{name}.ncnn.bin"))
    for index in range(40):
        directory = root / f"artifacts/llm_layer{index}_three_part"
        for part in (
            "part_a_attention_input",
            "part_b_attention_output",
            "part_c_mlp",
        ):
            files.extend(
                (
                    directory / part / f"{part}.ncnn.param",
                    directory / part / f"{part}.ncnn.bin",
                )
            )
    final_head = root / "artifacts/llm_final_head_ncnn"
    files.extend(
        (
            final_head / "final_norm/final_norm.ncnn.param",
            final_head / "final_norm/final_norm.ncnn.bin",
        )
    )
    for index in range(17):
        name = f"lm_head_shard_{index:02d}"
        files.extend(
            (
                final_head / name / f"{name}.ncnn.param",
                final_head / name / f"{name}.ncnn.bin",
            )
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.model_root.expanduser().resolve()
    expected = model_files(root)
    missing = [str(path.relative_to(root)) for path in expected if not path.is_file()]
    total_bytes = sum(path.stat().st_size for path in expected if path.is_file())
    report = {
        "status": "PASS" if not missing else "FAIL",
        "model_root": str(root),
        "expected_file_count": len(expected),
        "present_file_count": len(expected) - len(missing),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "missing": missing,
    }
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
