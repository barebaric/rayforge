"""
Runtime utilities for UI scripts.

This module provides the execution environment for scripts run via
`--uiscript`.
Scripts can explicitly import the app and window instances:

    from rayforge.uiscript import app, win
"""

import logging
import sys
import threading
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

app = None
win = None


def _set_context(application, window):
    """Called by the app to populate the script context."""
    global app, win
    app = application
    win = window


def run_script(script_path: Path, application, window):
    """
    Execute a UI script in a background thread.

    Args:
        script_path: Path to the Python script to execute.
        application: The RayforgeApplication instance.
        window: The MainWindow instance.
    """
    if not script_path.exists():
        logger.error(f"UIScript not found: {script_path}")
        return

    logger.info(f"Executing UI script: {script_path}")

    def execute():
        _set_context(application, window)

        script_globals = {
            "__name__": "__uiscript__",
            "__file__": str(script_path),
        }
        script_dir = str(script_path.parent.resolve())
        sys.path.insert(0, script_dir)
        try:
            with open(script_path, "r") as f:
                code = compile(f.read(), str(script_path), "exec")
            exec(code, script_globals)
        except Exception as e:
            logger.error(f"Error executing UI script: {e}")
            traceback.print_exc()
        finally:
            if sys.path[0] == script_dir:
                sys.path.pop(0)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
