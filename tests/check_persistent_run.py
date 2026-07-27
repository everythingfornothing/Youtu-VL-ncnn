#!/usr/bin/env python3
"""Validate a persistent runtime report and its generated token ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--official-tokens", type=Path, required=True)
    parser.add_argument("--expected-model-loads", type=int, default=138)
    args = parser.parse_args()

    runtime_path = args.run_dir / "runtime_report.json"
    generated_path = args.run_dir / "generated_token_ids.npy"
    report = json.loads(runtime_path.read_text(encoding="utf-8"))
    generated = np.load(generated_path, allow_pickle=False).reshape(-1)
    official = np.load(args.official_tokens, allow_pickle=False).reshape(-1)

    checks = {
        "persistent_mode": report.get("weight_mode") == "persistent",
        "resident_model_count": (
            report.get("resident_models") == args.expected_model_loads
        ),
        "single_model_load": (
            report.get("model_load_count") == args.expected_model_loads
        ),
        "reported_token_count": report.get("generated_tokens") == len(generated),
        "token_ids_exact": np.array_equal(generated, official),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "generated_tokens": len(generated),
        "model_load_count": report.get("model_load_count"),
        "total_seconds": report.get("total_seconds"),
        "decode_seconds": report.get("decode_seconds"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
