"""Download and sync third-party API docs from GitHub source archives.

Reads the pinned version of each package from requirements.txt, downloads the
matching source tarball from GitHub, and syncs its docs/api/ directory into the
website tree.
"""

import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REQUIREMENTS = Path("requirements.txt")


@dataclass(frozen=True)
class ApiDocsPackage:
    name: str
    repo_url: str
    output_dir: Path
    fallback_ref: str | None = None


PACKAGES = (
    ApiDocsPackage(
        name="raygeo",
        repo_url="https://github.com/barebaric/raygeo",
        output_dir=Path("website/docs/developer/raygeo-api"),
    ),
)


def _get_version(package: str) -> str:
    text = REQUIREMENTS.read_text()
    match = re.search(
        rf"^{re.escape(package)}==" + r"([^\s#]+)", text, re.MULTILINE
    )
    if not match:
        print(
            f"Could not find pinned {package} version in {REQUIREMENTS}.",
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
        if rel not in {p.relative_to(src) for p in src_files} and (
            dst_path.is_file()
        ):
            dst_path.unlink()


def _extract_docs(package: ApiDocsPackage, ref: str, tmp: str) -> Path:
    kind = "tags" if ref.startswith("v") else "heads"
    tar_url = f"{package.repo_url}/archive/refs/{kind}/{ref}.tar.gz"
    archive_path = Path(tmp) / f"{package.name}-{ref}.tar.gz"

    print(f"Downloading {package.name} {ref} source...")
    urllib.request.urlretrieve(tar_url, archive_path)

    print(f"Extracting {package.name} docs/api from archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=tmp, filter="data")

    prefix = f"{package.name}-{ref.removeprefix('v')}"
    return Path(tmp) / prefix / "docs" / "api"


def _sync_package(package: ApiDocsPackage) -> int:
    version = _get_version(package.name)
    ref = f"v{version}"

    with tempfile.TemporaryDirectory() as tmp:
        docs_src = _extract_docs(package, ref, tmp)

        if not docs_src.exists() and package.fallback_ref is not None:
            print(
                f"docs/api/ not found in {package.name} {ref}; "
                f"using {package.fallback_ref} docs instead.",
                file=sys.stderr,
            )
            docs_src = _extract_docs(package, package.fallback_ref, tmp)

        if not docs_src.exists():
            print(
                f"docs/api/ not found in the {package.name} {ref} archive.",
                file=sys.stderr,
            )
            return 1

        if not _needs_update(docs_src, package.output_dir):
            print(f"{package.name} API docs are up to date.")
            return 0

        print(f"Syncing {package.name} API docs...")
        _sync_dir(docs_src, package.output_dir)

    return 0


def main() -> int:
    for package in PACKAGES:
        if _sync_package(package):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
