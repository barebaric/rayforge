"""
Runtime utilities for early-startup scripts.

This module provides the execution environment for scripts run via
``--script``.  Unlike ``--uiscript`` (which runs after the main window
is mapped), ``--script`` runs *before* the application's addons are
loaded and before the main window is created.  This lets the script
register plugins with the ``pluggy`` plugin manager, configure
services, or influence environment variables.

Scripts can access the application context directly::

    from rayforge.context import get_context

    ctx = get_context()
    ctx.plugin_mgr.register(my_plugin)
"""

import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


def run_script(script_path: Path) -> None:
    """Execute an early-startup script synchronously.

    The script runs in the main thread before the application's
    addons are loaded, so it can register plugins, services, or
    configure the context.  If the script raises, the exception is
    logged and the application continues starting (the script is
    best-effort and should not block normal startup).

    Args:
        script_path: Path to the Python script to execute.
    """
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return

    logger.info(f"Executing startup script: {script_path}")

    script_globals = {
        "__name__": "__startup_script__",
        "__file__": str(script_path),
    }
    script_dir = str(script_path.parent.resolve())
    sys.path.insert(0, script_dir)
    try:
        with open(script_path, "r") as f:
            code = compile(f.read(), str(script_path), "exec")
        exec(code, script_globals)  # noqa: S102
    except Exception as e:  # noqa: BLE001 - arbitrary user script
        logger.error(f"Error executing startup script: {e}")
        traceback.print_exc()
    finally:
        if sys.path[0] == script_dir:
            sys.path.pop(0)
