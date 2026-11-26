#!/usr/bin/env python3
"""Render the MSIX application manifest for a release."""

import argparse
import re
import sys
from pathlib import Path
from string import Template
from xml.sax.saxutils import escape

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$")
_MAX_COMPONENT = 65535


def to_msix_version(version: str) -> str:
    """Converts a release version to a four-part MSIX version."""
    match = _VERSION_RE.fullmatch(version.strip())
    if not match:
        raise ValueError(
            f"MSIX requires a stable numeric version, got {version!r}"
        )
    parts = [int(part) for part in match.groups() if part is not None]
    if len(parts) == 3:
        parts.append(0)
    if parts[0] < 1:
        raise ValueError("The major MSIX version must be at least 1")
    if any(part > _MAX_COMPONENT for part in parts):
        raise ValueError(
            f"MSIX version components must be <= {_MAX_COMPONENT}"
        )
    return ".".join(str(part) for part in parts)


def render_template(template: str, values: dict[str, str]) -> str:
    """Substitutes safely escaped values into the manifest template."""
    escaped = {
        key: escape(value, {'"': "&quot;"}) for key, value in values.items()
    }
    return Template(template).substitute(escaped)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the MSIX application manifest."
    )
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--identity-name", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--publisher-display-name", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        version = to_msix_version(args.version)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    template = args.template.read_text(encoding="utf-8")
    output = render_template(
        template,
        {
            "version": version,
            "executable": args.executable,
            "identity_name": args.identity_name,
            "publisher": args.publisher,
            "publisher_display_name": args.publisher_display_name,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
