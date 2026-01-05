import dataclasses
import json
import base64
import yaml
import logging
import tempfile
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class DebugLogEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle types used in LogEntry, such as bytes,
    datetimes, enums, dataclasses, and logging.LogRecord.
    """

    def default(self, o):
        # Local import to avoid circular dependency at module load time.
        from .machine.driver.driver import DeviceState

        if isinstance(o, logging.LogRecord):
            # Sanitize the 'extra' dictionary in the LogRecord to prevent
            # serialization errors from non-JSON-compliant types attached by
            # third-party libraries (like pyvips' FFILibrary object).

            # Get keys that are in the instance's dict but not in the base
            # class. These are the keys that were added via the `extra` kwarg.
            extra_keys = set(o.__dict__.keys()) - set(
                logging.LogRecord.__dict__.keys()
            )
            safe_extra = {}
            for key in extra_keys:
                value = o.__dict__[key]
                # If the value is not a basic serializable type, convert it to
                # a string representation to ensure the dump doesn't fail.
                if not isinstance(
                    value, (str, int, float, bool, type(None), list, dict)
                ):
                    safe_extra[key] = repr(value)
                else:
                    safe_extra[key] = value

            # Format LogRecord into a serializable dictionary
            return {
                "created": o.created,
                "name": o.name,
                "levelname": o.levelname,
                "message": o.getMessage(),
                "exc_text": o.exc_text,
                "extra": safe_extra,  # Use the sanitized dictionary
            }
        if isinstance(o, DeviceState):
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


class DebugDumpManager:
    """
    Orchestrates the creation of comprehensive debug dump files using the
    new logging system.
    """

    def create_dump_archive(self) -> Optional[Path]:
        """
        Gathers all debug information, writes it to a temporary directory,
        and creates a ZIP archive.
        """
        # Perform imports locally to avoid circular dependencies at startup
        from .context import get_context
        from .ui_gtk.about import get_dependency_info
        from .logging_setup import get_memory_handler
        from .config import LOG_DIR
        from . import __version__

        logger.info("Creating debug dump archive...")
        try:
            context = get_context()
            config = context.config
            machine_mgr = context.machine_mgr
            memory_handler = get_memory_handler()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                # 1. Write in-memory log records to log_records.json
                if memory_handler:
                    log_snapshot = memory_handler.buffer
                    with open(tmp_path / "log_records.json", "w") as f:
                        json.dump(
                            log_snapshot, f, cls=DebugLogEncoder, indent=2
                        )
                else:
                    logger.warning(
                        "MemoryHandler not found, skipping record dump."
                    )

                # 2. Copy the latest session log file
                session_logs = sorted(
                    LOG_DIR.glob("session-*.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if session_logs:
                    latest_log = session_logs[0]
                    shutil.copy(latest_log, tmp_path / latest_log.name)
                else:
                    logger.warning(
                        "No session log file found to include in dump."
                    )

                # 3. Write system info to system_info.txt
                dep_info = get_dependency_info()
                with open(tmp_path / "system_info.txt", "w") as f:
                    f.write(f"## Rayforge {__version__ or 'Unknown'}\n\n")
                    for category, deps in dep_info.items():
                        f.write(f"### {category}\n")
                        for name, ver in deps:
                            f.write(f"{name}: {ver}\n")
                        f.write("\n")

                # 4. Write configs to YAML files
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

                # 5. Create ZIP archive
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
