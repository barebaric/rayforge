"""Linux V4L persistent device identification.

On Linux, /dev/videoN device numbers are assigned by the kernel based on
discovery order and can change between reboots. This module resolves
cameras using persistent /dev/v4l/by-id/ symlinks instead.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

V4L_BY_ID_DIR = Path("/dev/v4l/by-id")


def _is_video_capture_symlink(name: str) -> bool:
    """Check if a by-id symlink refers to a video capture device.

    Video capture symlinks end with '-video-indexN'. Other entries
    (like '-inputN' or '-altN') refer to non-capture interfaces.
    """
    return "-video-index" in name


def scan_v4l_by_id() -> Dict[str, str]:
    """Scan /dev/v4l/by-id/ for video capture devices.

    Returns:
        Dict mapping resolved '/dev/videoN' to the persistent
        '/dev/v4l/by-id/...' symlink path.
    """
    if not sys.platform.startswith("linux"):
        return {}

    if not V4L_BY_ID_DIR.is_dir():
        logger.debug("/dev/v4l/by-id/ does not exist")
        return {}

    result: Dict[str, str] = {}

    try:
        for entry in sorted(V4L_BY_ID_DIR.iterdir()):
            if not entry.is_symlink():
                continue
            if not _is_video_capture_symlink(entry.name):
                continue

            try:
                resolved = str(entry.resolve())
                if not resolved.startswith("/dev/video"):
                    continue
                if resolved not in result:
                    result[resolved] = str(entry)
            except OSError:
                continue
    except OSError as e:
        logger.debug(f"Error scanning /dev/v4l/by-id/: {e}")

    logger.debug(f"Found {len(result)} V4L by-id devices")
    return result


def get_sorted_by_id_paths() -> List[str]:
    """Return by-id paths sorted by underlying /dev/videoN number."""
    mapping = scan_v4l_by_id()

    def sort_key(by_id_path: str) -> int:
        resolved = Path(by_id_path).resolve()
        try:
            return int(resolved.name.replace("video", ""))
        except ValueError:
            return 999

    return sorted(mapping.values(), key=sort_key)


def resolve_device_id(device_id: str) -> str:
    """Resolve a numeric device ID to a persistent by-id path.

    If the device_id is already a by-id path or not a plain integer
    string, it is returned unchanged. If it is a numeric string like
    "0" on Linux, attempts to find the corresponding by-id path.

    Returns:
        The best available device identifier string.
    """
    if not sys.platform.startswith("linux"):
        return device_id

    if not device_id.isdigit():
        return device_id

    mapping = scan_v4l_by_id()
    target = f"/dev/video{device_id}"

    if target in mapping:
        logger.info(
            f"Migrated camera device_id from '{device_id}' "
            f"to '{mapping[target]}'"
        )
        return mapping[target]

    logger.debug(
        f"No by-id path found for {target}, keeping numeric ID"
    )
    return device_id


def friendly_name_from_by_id(by_id_path: str) -> str:
    """Extract a human-readable name from a by-id path.

    E.g. '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_1234'
    -> 'Logitech Webcam C930e'
    """
    filename = Path(by_id_path).stem

    for prefix in ("usb-", "pci-"):
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break

    idx = filename.find("-video-index")
    if idx >= 0:
        filename = filename[:idx]

    parts = filename.split("_")
    if len(parts) <= 1:
        return parts[0] if parts else ""

    # First segment is the hex vendor ID (e.g. '046d'), drop it.
    # Last segment is the serial number, drop it.
    if len(parts) > 2:
        parts = parts[1:-1]
    else:
        parts = parts[:-1]

    return " ".join(parts)
