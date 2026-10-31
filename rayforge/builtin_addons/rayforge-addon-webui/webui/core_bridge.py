"""
Bridge between the uvicorn event loop and the core TaskManager loop.

Rayforge's document model, drivers, and controllers live on the
TaskManager's asyncio loop (see :mod:`rayforge.serve`). The web server
runs on its own uvicorn loop in a different thread. Every DocEditor or
Machine mutation triggered by a route handler must therefore be
scheduled onto the TaskManager loop; blinker ``Signal.send`` is
synchronous, so touching shared state from the wrong thread races with
drivers running on their own threads.

Usage:

    result = await bridge.run(editor.file.save_project_to_path, path)
"""

import asyncio
import inspect
import logging

from rayforge.shared import tasker

logger = logging.getLogger(__name__)


class CoreBridge:
    """Schedules callables onto the TaskManager asyncio loop."""

    def __init__(self, editor, context):
        self._editor = editor
        self._context = context

    @property
    def editor(self):
        return self._editor

    @property
    def context(self):
        return self._context

    @property
    def core_loop(self) -> asyncio.AbstractEventLoop:
        """The TaskManager's asyncio loop (a different thread)."""
        return tasker.task_mgr.loop

    async def run(self, callable_, *args, **kwargs):
        """
        Run ``callable_(*args, **kwargs)`` on the TaskManager loop and
        await its result on the caller's loop.

        Sync callables run directly on the core loop; coroutine
        functions (or callables returning an awaitable) are awaited
        there. Exceptions propagate to the awaiting route handler.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._execute(callable_, *args, **kwargs), self.core_loop
        )
        return await asyncio.wrap_future(future)

    async def _execute(self, callable_, *args, **kwargs):
        result = callable_(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def schedule(self, callback, *args, **kwargs) -> None:
        """
        Fire-and-forget: invoke ``callback`` on the core loop without
        waiting for completion. Errors are logged, never raised.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._execute(callback, *args, **kwargs), self.core_loop
        )

        def _log_error(done) -> None:
            error = done.exception()
            if error is not None:
                logger.error(
                    "Core bridge task failed: %s", error, exc_info=error
                )

        future.add_done_callback(_log_error)
