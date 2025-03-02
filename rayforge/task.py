import asyncio
from asyncio.exceptions import CancelledError
from blinker import Signal
from typing import Optional, Callable, Coroutine
import threading
from .util.glib import idle_add


class Task:
    def __init__(self, coro, key=None, monitor_callback=None):
        self.coro = coro
        self.key = key if key is not None else id(self)
        self.monitor_callback = monitor_callback
        self._task = None
        self._status = "pending"
        self._progress = 0.0
        self.status_changed = Signal()
        self.progress_changed = Signal()

    async def run(self):
        """Run the task and update its status and progress."""
        self._status = "running"
        self._task = asyncio.create_task(self.coro)
        self._emit_status_changed()
        if self.monitor_callback:
            asyncio.create_task(self._monitor_progress())
        try:
            await self._task
            self._status = "completed"
            self._progress = 1.0
        except asyncio.CancelledError:
            self._status = "canceled"
        except Exception:
            self._status = "failed"
        finally:
            self._emit_status_changed()
            self._emit_progress_changed()

    async def _monitor_progress(self):
        """Monitor progress using the provided callback."""
        while not self._task.done():
            if self.monitor_callback:
                self._progress = self.monitor_callback(self)
                self._emit_progress_changed()
            await asyncio.sleep(0.1)

    def _emit_status_changed(self):
        """Emit status_changed signal from the main thread."""
        self.status_changed.send(self)

    def _emit_progress_changed(self):
        """Emit progress_changed signal from the main thread."""
        self.progress_changed.send(self)

    def get_status(self):
        """Get the current status of the task."""
        return self._status

    def get_progress(self):
        """Get the current progress of the task."""
        return self._progress

    def result(self):
        return self._task.result()

    def cancel(self):
        """Cancel the task."""
        if self._task and not self._task.done():
            self._task.cancel()


class TaskManager:
    def __init__(self):
        self._tasks = {}
        self.overall_progress_changed = Signal()
        self.running_tasks_changed = Signal()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            args=(self._loop,),
            daemon=True
        )
        self._thread.start()

    def _run_event_loop(self, loop):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def add_task(self, task: Task, when_done: Optional[Callable] = None):
        """Add a task to the manager."""
        old_task = self._tasks.get(task.key)
        if old_task:
            old_task.cancel()
        self._tasks[task.key] = task
        task.status_changed.connect(self._on_task_status_changed)
        task.progress_changed.connect(self._on_task_progress_changed)

        # Emit signals immediately when a new task is added
        self._emit_overall_progress_changed()
        self._emit_running_tasks_changed()

        asyncio.run_coroutine_threadsafe(
            self._run_task(task, when_done),
            self._loop
        )

    def add_coroutine(
        self,
        coro: Coroutine,
        key=None,
        monitor_callback=None,
        when_done: Optional[Callable] = None
    ):
        """
        Add a raw coroutine to the manager.
        The coroutine will be wrapped in a Task object internally.
        """
        task = Task(coro, key=key, monitor_callback=monitor_callback)
        self.add_task(task, when_done)

    async def _run_task(self, task: Task, when_done: Optional[Callable]):
        """Run the task and clean up when done."""
        try:
            await task.run()
        except CancelledError:
            pass
        finally:
            self._cleanup_task(task)
            self._emit_overall_progress_changed()
            self._emit_running_tasks_changed()
            if when_done:
                idle_add(when_done, task)

    def _cleanup_task(self, task: Task):
        """Clean up a completed task."""
        if task.key in self._tasks:
            del self._tasks[task.key]

    def _on_task_status_changed(self, task):
        """Handle task status changes."""
        self._emit_overall_progress_changed()
        self._emit_running_tasks_changed()

    def _on_task_progress_changed(self, task):
        """Handle task progress changes."""
        self._emit_overall_progress_changed()

    def _emit_overall_progress_changed(self):
        """Emit overall_progress_changed signal from the main thread."""
        progress = self.get_overall_progress()
        idle_add(
            self.overall_progress_changed.send,
            self,
            progress=progress
        )

    def _emit_running_tasks_changed(self):
        """Emit running_tasks_changed signal from the main thread."""
        idle_add(
            self.running_tasks_changed.send,
            self,
            tasks=list(self._tasks.values())
        )

    def get_overall_progress(self):
        """Calculate the overall progress of all tasks."""
        if not self._tasks:
            return 1.0
        total_progress = sum(
            task.get_progress() for task in self._tasks.values()
        )
        return total_progress / len(self._tasks)

    def shutdown(self):
        """Cancel all tasks and stop the event loop."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


task_mgr = TaskManager()
