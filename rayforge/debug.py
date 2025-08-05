import collections
import dataclasses
import json
import base64
import yaml
import logging
import threading
import tempfile
import shutil
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Deque, Optional, Dict

logger = logging.getLogger(__name__)

DEFAULT_LOG_BUFFER_SIZE = 5000


class LogType(Enum):
    """Enumeration for different types of log entries."""

    TX = auto()  # Raw data transmitted to the device
    RX = auto()  # Raw data received from the device
    DRIVER_CMD = auto()  # A high-level command issued by a driver
    STATE_CHANGE = auto()  # A device state change event
    APP_INFO = auto()  # General application log message
    ERROR = auto()  # An error or exception


@dataclasses.dataclass
class LogEntry:
    """A structured entry for the debug log."""

    timestamp: datetime
    source: str
    log_type: LogType
    data: Any


class DebugLogEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle types used in LogEntry, such as bytes,
    datetimes, enums, and dataclasses.
    """

    def default(self, o):
        # Local import to avoid circular dependency at module load time.
        from .driver.driver import DeviceState

        if isinstance(o, LogEntry) or isinstance(o, DeviceState):
            return dataclasses.asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, bytes):
            # Try to decode as UTF-8 for readability,
            # otherwise fall back to Base64
            try:
                return f"utf-8: '{o.decode('utf-8')}'"
            except UnicodeDecodeError:
                return f"b64: '{base64.b64encode(o).decode('ascii')}'"
        if isinstance(o, Exception):
            return repr(o)
        return super().default(o)


class DebugLogManager:
    """
    Manages a centralized, in-memory log buffer and orchestrates the creation
    of comprehensive debug dump files.
    """

    def __init__(self, max_entries: int = DEFAULT_LOG_BUFFER_SIZE):
        self._log_buffer: Deque[LogEntry] = collections.deque(
            maxlen=max_entries
        )
        self._lock = threading.Lock()

    def add_entry(
        self,
        source: str,
        log_type: LogType,
        data: Any,
    ):
        """
        Adds a new entry to the circular log buffer in a thread-safe manner.

        Args:
            source: The component originating the log (e.g., "GrblDriver").
            log_type: The category of the log entry (e.g., LogType.TX).
            data: The payload of the log entry (can be str, bytes, dict, etc.).
        """
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            source=source,
            log_type=log_type,
            data=data,
        )
        with self._lock:
            self._log_buffer.append(entry)

    def _get_log_snapshot(self) -> List[LogEntry]:
        """Returns a thread-safe copy of the current log buffer."""
        with self._lock:
            return list(self._log_buffer)

    def create_dump_archive(self) -> Optional[Path]:
        """
        Gathers all debug information, writes it to a temporary directory,
        and creates a ZIP archive.

        Returns:
            The Path to the created ZIP archive, or None on failure.
        """
        # Perform imports locally to avoid circular dependencies at startup
        from .config import config, machine_mgr
        from .shared.ui.about import get_dependency_info
        from . import __version__

        logger.info("Creating debug dump archive...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                # 1. Write logs to log.json
                log_snapshot = self._get_log_snapshot()
                with open(tmp_path / "log.json", "w") as f:
                    json.dump(log_snapshot, f, cls=DebugLogEncoder, indent=2)

                # 2. Write system info to system_info.txt
                dep_info = get_dependency_info()
                with open(tmp_path / "system_info.txt", "w") as f:
                    f.write(f"## Rayforge {__version__ or 'Unknown'}\n\n")
                    for category, deps in dep_info.items():
                        f.write(f"### {category}\n")
                        for name, ver in deps:
                            f.write(f"{name}: {ver}\n")
                        f.write("\n")

                # 3. Write configs to YAML files
                if config and config.machine:
                    with open(tmp_path / "active_machine.yaml", "w") as f:
                        yaml.safe_dump(config.machine.to_dict(), f)
                with open(tmp_path / "app_config.yaml", "w") as f:
                    yaml.safe_dump(config.to_dict(), f)

                all_machines_dict: Dict[str, Dict[str, Any]] = {
                    machine_id: machine.to_dict()
                    for machine_id, machine in machine_mgr.machines.items()
                }
                with open(tmp_path / "all_machines.yaml", "w") as f:
                    yaml.safe_dump(all_machines_dict, f)

                # 4. Create ZIP archive
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                archive_name = f"rayforge_debug_{timestamp_str}"
                # Use a system-wide temp dir for the final archive to ensure
                # it survives the 'with' block of the temporary directory.
                final_archive_base = Path(tempfile.gettempdir()) / archive_name

                shutil.make_archive(
                    str(final_archive_base), "zip", root_dir=tmpdir
                )
                archive_path = final_archive_base.with_suffix(".zip")
                logger.info(f"Debug dump archive created at {archive_path}")
                return archive_path

        except Exception:
            logger.error("Failed to create debug dump archive", exc_info=True)
            return None


# Create a singleton instance for global use
debug_log_manager = DebugLogManager()
