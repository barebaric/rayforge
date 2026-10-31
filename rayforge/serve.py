# flake8: noqa: E402
"""
Headless web server entry point.

Boots a GTK-free rayforge environment and serves the web UI provided
by an addon service (see the ``webui`` builtin addon). Run with:

    rayforge-serve --host 127.0.0.1 --port 8080
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import cast

# Parse --config early before any rayforge imports, as they may
# import config.py which computes CONFIG_DIR at module load time
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        os.environ["RAYFORGE_CONFIG_DIR"] = sys.argv[i + 1]
        break

import uvicorn

from rayforge.core.service_registry import service_registry
from rayforge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SERVICE_KEY = "webui"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rayforge-serve",
        description="Run the rayforge web UI without a display server.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST}). Use 0.0.0.0 to "
        "expose the server to the network; set --token in that case.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Require this shared token (Authorization: Bearer <token>).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration directory (defaults to the standard "
        "rayforge config directory).",
    )
    parser.add_argument(
        "--loglevel",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def _create_scheduler():
    from rayforge.shared import tasker

    def schedule_on_main_thread(callback, *args, **kwargs):
        tasker.task_mgr.loop.call_soon_threadsafe(
            lambda: callback(*args, **kwargs)
        )

    return schedule_on_main_thread


def _boot() -> tuple:
    """
    Creates the headless environment: context, task manager, and
    document editor. Addons are loaded with their worker and service
    entry points only.
    """
    from rayforge import config
    from rayforge.context import get_context
    from rayforge.doceditor.editor import DocEditor
    from rayforge.shared import tasker
    from rayforge.shared.tasker.manager import TaskManagerProxy
    from rayforge.worker_init import initialize_worker

    # Get the context first to ensure the ArtifactStore is created
    # before the TaskManager is initialized. This breaks the circular
    # dependency chain (app -> config -> machine -> task_manager).
    context = get_context()

    # Initialize the TaskManager with the worker initializer.
    # This MUST happen before accessing addon_mgr because the
    # MachineManager creates machines which import task_mgr, which
    # would trigger the creation of the TaskManager.
    task_mgr_proxy = cast(TaskManagerProxy, tasker.task_mgr)
    task_mgr_proxy.initialize(
        worker_initializer=initialize_worker,
        main_thread_scheduler=_create_scheduler(),
    )

    context.initialize_lite_context(config.MACHINE_DIR)

    # Accessing addon_mgr loads the addons (worker and service entry
    # points) and runs their registration hooks, which publishes the
    # web app factory into the global service registry.
    _ = context.addon_mgr

    editor = DocEditor(task_manager=tasker.task_mgr, context=context)
    return context, editor


def _shutdown(context, editor) -> None:
    from rayforge.shared import tasker

    async def shutdown_async():
        logger.info("Starting graceful async shutdown...")
        await context.shutdown()
        editor.cleanup()
        logger.info("Async shutdown complete.")

    task_mgr = tasker.task_mgr
    loop = task_mgr.loop
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(shutdown_async(), loop)
        try:
            future.result(timeout=10)
        except Exception:
            logger.exception("Error during graceful shutdown")
    else:
        logger.warning(
            "Task manager loop not running, skipping async shutdown."
        )

    if context.config_mgr:
        context.config_mgr.save()

    logger.info("Shutting down TaskManager")
    task_mgr.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(args.loglevel)

    context, editor = _boot()

    factory = service_registry.get(SERVICE_KEY)
    if factory is None:
        logger.error(
            "No '%s' service found: the webui addon is not available "
            "or disabled.",
            SERVICE_KEY,
        )
        return 1

    app = factory(editor=editor, context=context, token=args.token)

    context.machine_mgr.initialize_connections()

    logger.info(
        "Serving rayforge web UI on http://%s:%d", args.host, args.port
    )
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.loglevel.lower(),
        )
    finally:
        _shutdown(context, editor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
