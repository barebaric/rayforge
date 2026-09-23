"""Atomic file persistence helpers.

Writes YAML/dict payloads to disk without ever leaving a truncated or
half-written file behind. The previous content is preserved as a
``.bak`` sidecar so that a file damaged by an external event (power
loss, disk-full) can still be recovered.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def serialize_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def atomic_write_yaml(
    path: Path,
    data: Any,
    keep_backup: bool = True,
) -> None:
    """Serialize ``data`` to YAML and atomically replace ``path``.

    The payload is fully rendered in memory before the destination is
    touched, written to a temporary file in the same directory, flushed
    and fsynced, then moved into place with :func:`os.replace`. Either
    the old or the new version of the file exists afterwards, never a
    truncated mix. When ``keep_backup`` is set, the previous version is
    retained next to the file with a ``.bak`` suffix.

    Raises whatever the serialization or filesystem calls raise; the
    original file is left untouched in that case.
    """
    content = serialize_yaml(data)
    tmp_path = path.with_name(path.name + ".tmp")
    backup_path = path.with_name(path.name + ".bak")
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup and path.exists():
        try:
            os.replace(path, backup_path)
        except OSError as e:
            logger.warning("Could not update backup copy of %s: %s", path, e)
    os.replace(tmp_path, path)


def load_yaml_with_backup(path: Path) -> tuple[Any | None, bool]:
    """Load YAML from ``path``, falling back to its ``.bak`` sidecar.

    Returns a ``(data, recovered)`` tuple. ``data`` is the parsed
    document, or ``None`` if the file is empty; ``recovered`` is
    ``True`` when the primary file could not be read and the backup
    was used instead. Raises ``FileNotFoundError`` when neither the
    file nor a readable backup exists, and propagates parse errors
    from the primary file so callers can decide whether to surface or
    recover them.
    """
    try:
        return _read_yaml(path), False
    except (OSError, yaml.YAMLError) as e:
        backup_path = path.with_name(path.name + ".bak")
        if not backup_path.exists():
            raise
        logger.warning(
            "Could not read %s (%s); trying backup %s",
            path,
            e,
            backup_path,
        )
        try:
            data = _read_yaml(backup_path)
        except (OSError, yaml.YAMLError) as backup_error:
            logger.error(
                "Backup %s is also unreadable: %s",
                backup_path,
                backup_error,
            )
            raise e
        logger.warning("Recovered settings from backup %s", backup_path)
        return data, True


def _read_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)
