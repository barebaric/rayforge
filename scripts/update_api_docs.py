"""Regenerate API docs only if stubs are newer than markdown output."""

import sys
from pathlib import Path

from stubs_to_markdown import generate  # type: ignore[import]

STUBS_DIR = Path("external/raygeo/python/raygeo")
OUTPUT_DIR = Path("website/docs/developer/raygeo-api")


def _newest_mtime(files: list[Path]) -> float:
    return max((f.stat().st_mtime for f in files if f.exists()), default=0)


def _find_files(dirs: list[Path], pattern: str) -> list[Path]:
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob(pattern))
    return files


def _needs_update(src_files: list[Path], out_files: list[Path]) -> bool:
    if not out_files:
        return True
    src_mtime = _newest_mtime(src_files)
    out_mtime = _newest_mtime(out_files)
    return src_mtime > out_mtime


def main() -> int:
    src_files = _find_files([STUBS_DIR], "*.pyi")
    if not src_files:
        print("No stub files found.", file=sys.stderr)
        return 1

    out_files = _find_files([OUTPUT_DIR], "*.md")

    if not _needs_update(src_files, out_files):
        print("API docs are up to date.")
        return 0

    print("Regenerating API docs...")
    generate(STUBS_DIR, OUTPUT_DIR, "raygeo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
