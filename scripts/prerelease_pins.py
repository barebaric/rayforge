"""List packages pinned to pre-release versions in pixi.toml.

Prints the name and version of every dependency whose exact pin
`==<version>` refers to a pre-release version, one pair per line.
Re-pinning such a package to the latest stable release would be a
downgrade, so callers keep its pin and advance it individually with
a version floor instead.

scripts/tighten_pins.py re-pins advanced packages to the solved
version afterwards.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "pixi.toml"
SECTION = re.compile(
    r"^\[(?:feature\.[^.]+\.)?(dependencies|pypi-dependencies|"
    r"dev-dependencies)\]"
)
PIN = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*) *= *"==([^"]+)"')
PRERELEASE = re.compile(r"(?:a|b|c|rc|alpha|beta|pre|preview|dev)\d")


def main() -> int:
    lines = MANIFEST.read_text().splitlines()
    section = ""
    for line in lines:
        if line.startswith("["):
            match = SECTION.match(line.strip())
            section = match.group(1) if match else ""
            continue
        if not section:
            continue
        match = PIN.match(line)
        if match and PRERELEASE.search(match.group(2)):
            print(match.group(1), match.group(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
