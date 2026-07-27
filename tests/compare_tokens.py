#!/usr/bin/env python3
"""Compare official PyTorch and ncnn generated token arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--ncnn", type=Path, required=True)
    parser.add_argument("--official-text", type=Path)
    parser.add_argument("--ncnn-text", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    official = np.load(args.official, allow_pickle=False).reshape(-1).astype(np.int64)
    ncnn = np.load(args.ncnn, allow_pickle=False).reshape(-1).astype(np.int64)
    common = min(len(official), len(ncnn))
    different = np.flatnonzero(official[:common] != ncnn[:common])
    matching_prefix = int(different[0]) if different.size else common
    report = {
        "status": "PASS" if np.array_equal(official, ncnn) else "FAIL",
        "official_length": len(official),
        "ncnn_length": len(ncnn),
        "matching_prefix": matching_prefix,
        "first_mismatch": None,
    }
    if different.size:
        index = int(different[0])
        report["first_mismatch"] = {
            "step": index,
            "official": int(official[index]),
            "ncnn": int(ncnn[index]),
        }
    if args.official_text and args.ncnn_text:
        official_text = args.official_text.read_text(encoding="utf-8").rstrip()
        ncnn_text = args.ncnn_text.read_text(encoding="utf-8").rstrip()
        report["text_exact"] = official_text == ncnn_text
        if not report["text_exact"]:
            report["status"] = "FAIL"
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
