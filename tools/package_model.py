#!/usr/bin/env python3
"""Copy only runtime-required model files and write SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from check_model_layout import model_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    expected = model_files(source)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        rendered = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing required model files:\n{rendered}")
    protected_roots = (source / "artifacts", source / "models")
    if output == source or any(
        output == protected or protected in output.parents
        for protected in protected_roots
    ):
        raise ValueError(
            "output root must not overwrite the source, artifacts, or models"
        )

    checksum_lines = []
    for path in expected:
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_stat = path.stat()
        destination_matches = False
        if destination.is_file():
            destination_stat = destination.stat()
            destination_matches = (
                destination_stat.st_size == source_stat.st_size
                and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
            )
        if not destination_matches:
            shutil.copy2(path, destination)
        checksum_lines.append(f"{sha256(destination)}  {relative.as_posix()}")

    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    print(f"packaged {len(expected)} files in {output}")


if __name__ == "__main__":
    main()
