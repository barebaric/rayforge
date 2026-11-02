"""Restore exact version pins in pixi.toml from the solved lock file.

Takes a file with package names, one per line, looks up the version
resolved for each package in pixi.lock, and rewrites the package's
version floor in the dependency tables back to an exact `==` pin.
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "pixi.toml"
LOCK = ROOT / "pixi.lock"
SECTION = re.compile(
    r"^\[(?:feature\.[^.]+\.)?(dependencies|pypi-dependencies|"
    r"dev-dependencies)\]"
)
FLOOR = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*) *= *">=?([^"]+)"')


def normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def parse_conda_url(url: str) -> tuple:
    stem = url.rsplit("/", 1)[-1].removesuffix(".conda")
    parts = stem.split("-")
    if len(parts) < 3:
        return "", ""
    return "-".join(parts[:-2]), parts[-2]


def lock_versions() -> dict:
    with LOCK.open() as handle:
        packages = yaml.safe_load(handle)["packages"]
    versions = {}
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        if name is None or version is None:
            name, version = parse_conda_url(package.get("conda", ""))
        if name and version:
            versions.setdefault(normalize(name), version)
    return versions


def main() -> int:
    with Path(sys.argv[1]).open() as handle:
        names = {normalize(name) for name in handle.read().split()}
    versions = lock_versions()
    lines = MANIFEST.read_text().splitlines()
    section = ""
    for index, line in enumerate(lines):
        if line.startswith("["):
            match = SECTION.match(line.strip())
            section = match.group(1) if match else ""
            continue
        if not section:
            continue
        match = FLOOR.match(line)
        if not match or normalize(match.group(1)) not in names:
            continue
        version = versions[normalize(match.group(1))]
        lines[index] = f'{match.group(1)} = "=={version}"'
    MANIFEST.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
