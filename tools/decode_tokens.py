#!/usr/bin/env python3
"""Decode generated Youtu-VL token ids without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


DEFAULT_MODEL = "tencent/Youtu-VL-4B-Instruct"
DEFAULT_REVISION = "8d30a0e49662a1d628a472b12df264dbcd768753"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    token_ids = np.load(args.tokens, allow_pickle=False).reshape(-1).astype(int).tolist()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).replace("\ufffd", "").replace("\x00", "").rstrip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    report = {
        "token_ids": token_ids,
        "text": text,
        "eos_token_id": tokenizer.eos_token_id,
        "reached_eos": bool(token_ids and token_ids[-1] == tokenizer.eos_token_id),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
