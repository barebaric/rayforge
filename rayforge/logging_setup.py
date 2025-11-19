import logging
import logging.handlers
import sys
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from blinker import Signal
from .config import LOG_DIR

_memory_handler_instance: Optional[logging.handlers.MemoryHandler] = None
_ui_formatter_instance: Optional[logging.Formatter] = None

LOG_FILES_TO_KEEP = 5

# Global signal for UI log events
ui_log_event_received = Signal()


class UILogFilter(logging.Filter):
    """
    This filter only allows log records that are intended for the user-facing
    log dialog, such as machine events, warnings, and errors.
    """

    UI_CATEGORIES = {"MACHINE_EVENT", "ERROR", "WARNING", "STATE_CHANGE"}

    def filter(self, record: logging.LogRecord) -> bool:
        return record.__dict__.get("log_category") in self.UI_CATEGORIES


class ConsoleLogFilter(logging.Filter):
    """
    This filter rejects log records that are categorized as 'RAW_IO'
    to prevent spamming the console with low-level communication data.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.__dict__.get("log_category") != "RAW_IO"


class UILogHandler(logging.Handler):
    """
    A custom logging handler that forwards filtered log records to the UI
    via a blinker signal.
    """

    def emit(self, record: logging.LogRecord):
        # We don't need to format the message here, as the dialog will add its
        # own timestamp. We just pass the core message.
        log_entry = self.format(record)
        ui_log_event_received.send(self, message=log_entry)


def _cleanup_old_logs(log_dir: Path, keep_count: int):
    """
    Deletes old log files, keeping only the most recent 'keep_count' number
    of logs.
    """
    try:
        # Find all log files that match our session pattern
        log_files: List[Path] = sorted(
            log_dir.glob("session-*.log"),
            key=lambda p: p.stat().st_mtime,  # Sort by modification time
            reverse=True,  # Newest first
        )

        # If we have more logs than we want to keep, delete the oldest ones
        if len(log_files) > keep_count:
            files_to_delete = log_files[keep_count:]
            logging.debug(
                f"Log cleanup: Deleting {len(files_to_delete)} old log files."
            )
            for f in files_to_delete:
                try:
                    f.unlink()
                except OSError as e:
                    logging.warning(f"Could not delete old log file {f}: {e}")
    except Exception as e:
        # We don't want a logging failure to crash the app startup
        logging.error(f"An unexpected error occurred during log cleanup: {e}")


def get_memory_handler() -> Optional[logging.handlers.MemoryHandler]:
    """
    Returns the global instance of the MemoryHandler.
    This is used by the DebugDumpManager to retrieve the log buffer.
    """
    return _memory_handler_instance


def get_ui_formatter() -> Optional[logging.Formatter]:
    """
    Returns the global instance of the Formatter used for the UI Log.
    """
    return _ui_formatter_instance


def setup_logging(loglevel_str: str):
    """
    Configures the root logger with console, file, and in-memory handlers.
    This replaces the need for logging.basicConfig().

    Args:
        loglevel_str: The desired logging level for the console as a string
                      (e.g., "INFO", "DEBUG").
    """
    global _memory_handler_instance, _ui_formatter_instance

    log_level = getattr(logging, loglevel_str.upper(), logging.INFO)
    root_logger = logging.getLogger()

    # Clear any handlers already configured (e.g., by basicConfig)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set root logger to the lowest level to capture everything.
    # Handlers will then filter messages by their own specific levels.
    root_logger.setLevel(logging.DEBUG)

    # 1. Add the console handler IMMEDIATELY after clearing.
    # This prevents logging calls in helper functions (like _cleanup_old_logs)
    # from implicitly re-triggering basicConfig.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ConsoleLogFilter())
    root_logger.addHandler(console_handler)

    # Now it is safe to run functions that might log things.
    _cleanup_old_logs(LOG_DIR, LOG_FILES_TO_KEEP)

    # 2. Session File Handler (for persistent, detailed logs)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"session-{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture max detail in file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(threadName)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 3. In-Memory Handler (for the debug dump feature)
    # We set a high flushLevel so it effectively never flushes on its own.
    # We will access its buffer directly when creating a debug dump.
    _memory_handler_instance = logging.handlers.MemoryHandler(
        capacity=5000, flushLevel=logging.CRITICAL + 1
    )
    _memory_handler_instance.setLevel(logging.DEBUG)
    root_logger.addHandler(_memory_handler_instance)

    # 4. UI Log Handler (for the MachineLogDialog)
    ui_handler = UILogHandler()
    ui_handler.setLevel(logging.INFO)  # Don't show DEBUG messages in UI log
    ui_handler.addFilter(UILogFilter())
    # Create the formatter and store it in our global instance
    _ui_formatter_instance = logging.Formatter(
        "[%(asctime)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ui_handler.setFormatter(_ui_formatter_instance)
    root_logger.addHandler(ui_handler)

    # Silence noisy third-party loggers
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("pyvips.vobject").setLevel(logging.WARNING)

    logging.info(f"Logging configured. Console level: {loglevel_str}")
    logging.info(f"Session log file: {log_file}")
    logging.debug("Silenced pyvips.vobject logger to WARNING level.")
