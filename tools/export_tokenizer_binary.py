#!/usr/bin/env python3
"""Export Hugging Face ByteLevel-BPE data for the dependency-free C++ runtime."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"YVLTOK2\0"


def bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


BYTE_DECODER = {character: value for value, character in bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    try:
        return bytes(BYTE_DECODER[character] for character in token)
    except KeyError as error:
        raise ValueError(f"base-vocabulary token is not ByteLevel: {token!r}") from error


def write_u32(stream, value: int) -> None:
    stream.write(struct.pack("<I", value))


def write_bytes(stream, value: bytes) -> None:
    write_u32(stream, len(value))
    stream.write(value)


def export(tokenizer_json: Path, output: Path) -> None:
    tokenizer = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = tokenizer["model"]
    if model["type"] != "BPE":
        raise ValueError("only a BPE tokenizer is supported")

    vocabulary = model["vocab"]
    base_count = len(vocabulary)
    by_id: list[str | None] = [None] * base_count
    for token, token_id in vocabulary.items():
        if token_id < 0 or token_id >= base_count or by_id[token_id] is not None:
            raise ValueError("base vocabulary IDs must be contiguous and unique")
        by_id[token_id] = token
    if any(token is None for token in by_id):
        raise ValueError("base vocabulary contains an ID gap")

    merges = model["merges"]
    added = tokenizer["added_tokens"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        write_u32(stream, base_count)
        write_u32(stream, len(merges))
        write_u32(stream, len(added))
        for token in by_id:
            assert token is not None
            write_bytes(stream, token_to_bytes(token))
        for merge in merges:
            if not isinstance(merge, list) or len(merge) != 2:
                raise ValueError(f"unexpected BPE merge: {merge!r}")
            write_bytes(stream, token_to_bytes(merge[0]))
            write_bytes(stream, token_to_bytes(merge[1]))
        for token in added:
            write_u32(stream, token["id"])
            stream.write(bytes([1 if token["special"] else 0]))
            write_bytes(stream, token["content"].encode("utf-8"))

    print(
        f"exported {base_count} base tokens, {len(merges)} merges, "
        f"{len(added)} added tokens to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.tokenizer_json, args.output)


if __name__ == "__main__":
    main()
