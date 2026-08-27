"""Journal of device identification events.

Every time discovered devices are matched against profiles, the
device's reported identity and the ranked candidate profiles are
recorded here. Only the most recent :data:`MAX_ENTRIES` entries are
kept; the file is rewritten in full on each record rather than
appended to, which is simpler than in-place trimming at this size.
Help → Save Debug Log includes the journal, so a misidentification
can be diagnosed after the fact even though the discovery itself was
transient.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..discovery import DeviceIdentity
from .matching import ProfileMatch

logger = logging.getLogger(__name__)

FILENAME = "device-identification.jsonl"

#: Oldest entries beyond this count are dropped on write.
MAX_ENTRIES = 500


def journal_file(config_dir: Path) -> Path:
    return Path(config_dir) / FILENAME


def record_match(
    path: Path,
    identity: DeviceIdentity,
    matches: list[ProfileMatch],
) -> None:
    """
    Record one identification event. Never raises: journaling must
    not break matching or the wizard.
    """
    vid_hex = (
        f"{identity.usb_vid:04x}" if identity.usb_vid is not None else None
    )
    pid_hex = (
        f"{identity.usb_pid:04x}" if identity.usb_pid is not None else None
    )
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "firmware": identity.firmware,
        "banner": identity.banner,
        "usb_vid": vid_hex,
        "usb_pid": pid_hex,
        "tokens": sorted(identity.tokens),
        "matches": [
            {"profile": m.profile.name, "confidence": m.confidence}
            for m in matches
        ],
    }
    _write(path, entry)


def read_entries(path: Path) -> list[dict]:
    """All journal entries, oldest first. Undecodable lines are
    skipped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping corrupted journal line")
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _write(path: Path, entry: dict) -> None:
    try:
        entries = read_entries(path)
        entries.append(entry)
        if len(entries) > MAX_ENTRIES:
            entries = entries[-MAX_ENTRIES:]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item) + "\n" for item in entries)
    except OSError as e:
        logger.warning(f"Could not write device journal to {path}: {e}")
