"""Download and sync third-party API docs from GitHub source archives.

Reads the pinned version or git revision of each package from requirements.txt,
downloads the matching source tarball from GitHub, and syncs its docs/api/
directory into the website tree.
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


@dataclass(frozen=True)
class PinnedRef:
    ref: str
    kind: str


REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"\s*(?:\[[^\]]*\])?\s*(?P<spec>==\S+|@\s*\S+)"
)

PACKAGES = (
    ApiDocsPackage(
        name="raygeo",
        repo_url="https://github.com/barebaric/raygeo",
        output_dir=Path("website/docs/developer/raygeo-api"),
    ),
    ApiDocsPackage(
        name="raydriver",
        repo_url="https://github.com/barebaric/raydriver",
        output_dir=Path("website/docs/developer/raydriver-api"),
        # The pinned release predates docs generation; use the default
        # branch until a release contains docs/api/.
        fallback_ref="main",
    ),
)


def _parse_git_ref(spec: str) -> PinnedRef:
    url = spec.removeprefix("git+").split("#", 1)[0]
    if "@" not in url:
        return PinnedRef("main", "branch")
    _, ref = url.rsplit("@", 1)
    if not ref:
        return PinnedRef("main", "branch")
    if ref.startswith("refs/tags/"):
        return PinnedRef(ref.removeprefix("refs/tags/"), "tag")
    if ref.startswith("refs/heads/"):
        return PinnedRef(ref.removeprefix("refs/heads/"), "branch")
    if re.fullmatch(r"[0-9a-f]{7,40}", ref):
        return PinnedRef(ref, "commit")
    if ref.startswith("v"):
        return PinnedRef(ref, "tag")
    return PinnedRef(ref, "branch")


def _get_pinned_ref(package: str) -> PinnedRef:
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = REQUIREMENT.match(line)
        if not match or match.group("name") != package:
            continue
        spec = match.group("spec")
        if spec.startswith("=="):
            return PinnedRef(f"v{spec.removeprefix('==')}", "tag")
        return _parse_git_ref(spec.removeprefix("@").strip())
    print(
        f"Could not find pinned {package} version in {REQUIREMENTS}.",
        file=sys.stderr,
    )
    sys.exit(1)


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


def _archive_url(package: ApiDocsPackage, pinned: PinnedRef) -> str:
    if pinned.kind == "commit":
        return f"{package.repo_url}/archive/{pinned.ref}.tar.gz"
    collection = "tags" if pinned.kind == "tag" else "heads"
    return f"{package.repo_url}/archive/refs/{collection}/{pinned.ref}.tar.gz"


def _extract_docs(
    package: ApiDocsPackage, pinned: PinnedRef, tmp: str
) -> Path:
    tar_url = _archive_url(package, pinned)
    safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "-", pinned.ref)
    archive_path = Path(tmp) / f"{package.name}-{safe_ref}.tar.gz"
    extract_dir = Path(tmp) / f"{package.name}-{safe_ref}"
    extract_dir.mkdir()

    print(f"Downloading {package.name} {pinned.ref} source...")
    urllib.request.urlretrieve(tar_url, archive_path)

    print(f"Extracting {package.name} docs/api from archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir, filter="data")

    roots = [path for path in extract_dir.iterdir() if path.is_dir()]
    if not roots:
        return extract_dir / "docs" / "api"
    return roots[0] / "docs" / "api"


def _sync_package(package: ApiDocsPackage) -> int:
    pinned = _get_pinned_ref(package.name)

    with tempfile.TemporaryDirectory() as tmp:
        docs_src = _extract_docs(package, pinned, tmp)

        if not docs_src.exists() and package.fallback_ref is not None:
            print(
                f"docs/api/ not found in {package.name} {pinned.ref}; "
                f"using {package.fallback_ref} docs instead.",
                file=sys.stderr,
            )
            fallback = PinnedRef(package.fallback_ref, "branch")
            docs_src = _extract_docs(package, fallback, tmp)

        if not docs_src.exists():
            print(
                f"docs/api/ not found in the {package.name} "
                f"{pinned.ref} archive.",
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
