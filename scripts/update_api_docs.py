"""Download and sync raygeo API docs from the GitHub source archive.

Reads the pinned raygeo version from requirements.txt, downloads the
matching source tarball from GitHub, and syncs docs/api/ into the
website tree.
"""

import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/barebaric/raygeo"
REQUIREMENTS = Path("requirements.txt")
OUTPUT_DIR = Path("website/docs/developer/raygeo-api")


def _get_raygeo_version() -> str:
    text = REQUIREMENTS.read_text()
    match = re.search(r"^raygeo==([\d.]+)", text, re.MULTILINE)
    if not match:
        print(
            f"Could not find pinned raygeo version in {REQUIREMENTS}.",
            file=sys.stderr,
        )
        sys.exit(1)
    return match.group(1)


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
    version = _get_raygeo_version()
    tar_url = f"{REPO_URL}/archive/refs/tags/v{version}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "raygeo.tar.gz"
        print(f"Downloading raygeo v{version} source...")
        urllib.request.urlretrieve(tar_url, archive_path)

        prefix = f"raygeo-{version}"
        docs_src = Path(tmp) / prefix / "docs" / "api"

        print("Extracting docs/api from archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmp, filter="data")

        if not docs_src.exists():
            print(
                f"docs/api/ not found in the v{version} archive.",
                file=sys.stderr,
            )
            return 1

        if not _needs_update(docs_src, OUTPUT_DIR):
            print("API docs are up to date.")
            return 0

        print("Syncing raygeo API docs...")
        _sync_dir(docs_src, OUTPUT_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
