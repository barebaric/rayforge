"""Copy pre-generated raygeo API docs from external/raygeo/docs/api/.

Raygeo now generates its own Docusaurus-compatible documentation.
This script simply syncs those files into the website tree.
"""

import shutil
import sys
from pathlib import Path

RAYGEO_DOCS = Path("external/raygeo/docs/api")
OUTPUT_DIR = Path("website/docs/developer/raygeo-api")


def _newest_mtime(files: list[Path]) -> float:
    return max((f.stat().st_mtime for f in files if f.exists()), default=0)


def _find_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.rglob(pattern)) if directory.exists() else []


def _needs_update(src_dir: Path, out_dir: Path) -> bool:
    src_files = _find_files(src_dir, "*.*")
    out_files = _find_files(out_dir, "*.*")
    if not out_files:
        return True
    return _newest_mtime(src_files) > _newest_mtime(out_files)


def _sync_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    src_files = set(_find_files(src, "*.*"))

    existing_dst_files = set(_find_files(dst, "*.*"))

    for src_path in src_files:
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            not dst_path.exists()
            or src_path.stat().st_mtime > dst_path.stat().st_mtime
        ):
            shutil.copy2(src_path, dst_path)

    for dst_path in existing_dst_files:
        rel = dst_path.relative_to(dst)
        if rel not in {p.relative_to(src) for p in src_files}:
            if dst_path.is_file():
                dst_path.unlink()


def main() -> int:
    if not RAYGEO_DOCS.exists():
        print(f"Raygeo docs not found at {RAYGEO_DOCS}.", file=sys.stderr)
        return 1

    if not _needs_update(RAYGEO_DOCS, OUTPUT_DIR):
        print("API docs are up to date.")
        return 0

    print("Syncing raygeo API docs...")
    _sync_dir(RAYGEO_DOCS, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
