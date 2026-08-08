"""Sync pip requirement files with the pins in pixi.toml.

pixi.toml is the single source of truth for dependency versions. This
script rewrites requirements.txt and debian/requirements-bundle.txt so
that every dependency that also exists in pixi.toml carries the same
version spec. Entries without a pixi.toml counterpart are left as-is.
"""

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
PIXI_TOML = ROOT / "pixi.toml"
REQUIREMENT_FILES = [
    ROOT / "requirements.txt",
    ROOT / "debian" / "requirements-bundle.txt",
]
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$")


def normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def load_specs() -> dict:
    with PIXI_TOML.open("rb") as handle:
        manifest = tomllib.load(handle)
    specs = {}
    features = manifest.get("feature", {})
    sections = [
        features.get("app", {}).get("pypi-dependencies", {}),
        features.get("app", {}).get("dependencies", {}),
        manifest.get("dependencies", {}),
    ]
    for section in sections:
        for name, spec in section.items():
            if isinstance(spec, str):
                specs.setdefault(normalize(name), spec)
    return specs


def sync_file(path: Path, specs: dict, exact: bool) -> None:
    text = path.read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = REQUIREMENT.match(line.strip())
        if not match:
            continue
        name = match.group(1)
        spec = specs.get(normalize(name))
        if spec is None or (exact and not spec.startswith("==")):
            continue
        new_line = f"{name}{spec}" if spec != "*" else name
        if new_line != line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}{new_line}"
    content = "\n".join(lines)
    if text.endswith("\n"):
        content += "\n"
    path.write_text(content)


def main() -> int:
    specs = load_specs()
    for path in REQUIREMENT_FILES:
        sync_file(path, specs, exact=path.name == "requirements-bundle.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
