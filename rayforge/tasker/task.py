"""
Task module for managing individual tasks.
"""

from __future__ import annotations
import asyncio
from asyncio.exceptions import CancelledError
import logging
from typing import Optional, Callable, Coroutine, Any
from blinker import Signal
from .context import ExecutionContext


logger = logging.getLogger(__name__)


class Task:
    def __init__(
        self,
        coro: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        key: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.coro = coro
        self.args = args
        self.kwargs = kwargs
        self.key: Any = key if key is not None else id(self)
        self._task: Optional[asyncio.Task[Any]] = None
        self._task_result: Any = None
        self._task_exception: Optional[BaseException] = None
        self._status: str = "pending"
        self._progress: float = 0.0
        self._message: Optional[str] = None
        self._cancel_requested: bool = False  # Flag for early cancellation
        self.status_changed: Signal = Signal()
        self.event_received: Signal = Signal()

    def update(
        self, progress: Optional[float] = None, message: Optional[str] = None
    ) -> None:
        """
        Updates task progress and/or message. This method is designed to be
        called from the main thread (e.g., via idle_add) and emits a
        single signal for any change.
        """
        updated = False
        if progress is not None and self._progress != progress:
            self._progress = progress
            updated = True
        if message is not None and self._message != message:
            self._message = message
            updated = True

        if updated:
            self._emit_status_changed()

    async def run(self, context: ExecutionContext) -> None:
        """
        Run the task and update its status. The wrapped coroutine is
        responsible for reporting progress via the provided context.
        """
        logger.debug(f"Task {self.key}: Entering run method.")

        # Early cancellation check
        if self._cancel_requested:
            logger.debug(
                f"Task {self.key}: Cancellation requested before coro start."
            )
            self._status = "canceled"
            self._emit_status_changed()
            raise CancelledError("Task cancelled before coro execution")

        # Start execution
        self._status = "running"
        self._emit_status_changed()  # Emit running status
        logger.debug(
            f"Task {self.key}: Creating internal asyncio.Task for coro."
        )

        # Wrap the coroutine in a Task.
        self._task = asyncio.create_task(
            self.coro(context, *self.args, **self.kwargs)
        )

        # Await Coroutine Completion
        try:
            logger.debug(f"Task {self.key}: Awaiting internal asyncio.Task.")
            await self._task
            # If await completes without CancelledError or other Exception:
            logger.debug(f"Task {self.key}: Coro completed successfully.")
            self._status = "completed"
            self._progress = 1.0
        except asyncio.CancelledError:
            # This catches cancellation of self._task (the coro)
            logger.warning(
                f"Task {self.key}: Internal asyncio.Task was cancelled."
            )
            self._status = "canceled"
            # Propagate so the outer _run_task knows about the cancellation
            raise
        except Exception:
            logger.exception(f"Task {self.key}: Coro failed with exception.")
            self._status = "failed"
            # Re-raise so the TaskManager can see and log it.
            raise
        finally:
            logger.debug(
                f"Task {self.key}: Run method finished "
                f"with status '{self._status}'."
            )
            # First, flush any pending context updates. This might call
            # self.update() and set an intermediate state (e.g. final message).
            context.flush()

            # Now, set the authoritative final state.
            if self._status == "completed":
                self._progress = 1.0

            # Emit one final signal with the authoritative state.
            self._emit_status_changed()

    def _emit_status_changed(self) -> None:
        """Emit status_changed signal from the main thread."""
        self.status_changed.send(self)

    def get_progress(self) -> float:
        """Get the current progress of the task."""
        return self._progress

    def get_status(self) -> str:
        """Get the current lifecycle status of the task."""
        return self._status

    def get_message(self) -> Optional[str]:
        """Get the current user-facing message for the task."""
        return self._message

    def result(self) -> Any:
        if self._task:  # It's an asyncio-managed task
            if not self._task.done():
                raise asyncio.InvalidStateError("result is not yet available")
            return self._task.result()

        # It's a synchronously-managed (process) task
        if self._status == "completed":
            return self._task_result
        if self._status == "failed":
            if self._task_exception:
                raise self._task_exception
            raise asyncio.InvalidStateError(
                "Task failed but no exception was captured."
            )
        if self._status == "canceled":
            raise CancelledError("Task was cancelled.")

        raise asyncio.InvalidStateError(
            f"result is not available for task in state '{self._status}'"
        )

    def cancel(self) -> None:
        """
        Request cancellation of the task.
        Sets a flag to prevent starting if not already started,
        and attempts to cancel the underlying asyncio.Task if it exists.
        """
        logger.debug(f"Task {self.key}: Cancel method called.")
        self._cancel_requested = True  # Set flag regardless of current state

        task_to_cancel = self._task
        if task_to_cancel and not task_to_cancel.done():
            logger.info(
                f"Task {self.key}: Attempting to cancel "
                f"running internal asyncio.Task."
            )
            task_to_cancel.cancel()
        elif task_to_cancel:
            logger.debug(
                f"Task {self.key}: Internal asyncio.Task already done."
            )
        else:
            logger.debug(
                f"Task {self.key}: Internal asyncio.Task not yet "
                f"created, flag set."
            )

    def is_cancelled(self) -> bool:
        """Checks if cancellation has been requested for this task."""
        return self._cancel_requested
