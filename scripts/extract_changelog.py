"""Print the changelog section of a release version.

Usage: extract_changelog.py VERSION [CHANGELOG]

VERSION is the release version or tag, with or without a leading "v".
The matching "## <version>" section is printed to stdout without its
heading and without the sections that follow it. Exits with status 1
when the changelog has no matching section.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHANGELOG = ROOT / "CHANGELOG.md"
HEADING = re.compile(r"^##\s+(.+?)\s*$")
DATE_SUFFIX = re.compile(r"\s+-\s+\d{4}-\d{2}-\d{2}$")


def normalize(text: str) -> str:
    """Turn a changelog heading or tag into a bare version string."""
    text = DATE_SUFFIX.sub("", text.strip())
    text = text.strip("[]").strip()
    return re.sub(r"^[vV](?=\d)", "", text)


def extract(text: str, version: str) -> str | None:
    """Return the body of the version's section, or None."""
    wanted = normalize(version)
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if match and normalize(match.group(1)) == wanted:
            start = index + 1
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start, len(lines)):
        if HEADING.match(lines[index]):
            end = index
            break
    section = "\n".join(lines[start:end]).strip()
    return section or None


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} VERSION [CHANGELOG]",
            file=sys.stderr,
        )
        return 2
    version = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_CHANGELOG
    section = extract(path.read_text(encoding="utf-8"), version)
    if section is None:
        print(
            f"No changelog section for {version!r} in {path}",
            file=sys.stderr,
        )
        return 1
    print(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
