from __future__ import annotations
import logging
import asyncio
from typing import TYPE_CHECKING, Optional, Callable, Coroutine
from blinker import Signal
from ..pipeline.artifact import JobArtifact, JobArtifactHandle
from ..context import get_context
from ..shared.util.glib import idle_add
from .job_monitor import JobMonitor

if TYPE_CHECKING:
    from .models.machine import Machine
    from .driver.driver import Axis
    from ..doceditor.editor import DocEditor
    from ..core.ops import Ops


logger = logging.getLogger(__name__)


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self.job_started = Signal()
        self._current_monitor: Optional[JobMonitor] = None
        self._on_progress_callback: Optional[Callable[[dict], None]] = None

    def home_machine(self, machine: "Machine"):
        """Adds a 'home' task to the task manager for the given machine."""
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.home(), key="home-machine"
        )

    def select_tool(self, machine: "Machine", head_index: int):
        """Adds a 'select_head' task to the task manager."""
        if not (0 <= head_index < len(machine.heads)):
            logger.error(f"Invalid head index {head_index} for tool selection")
            return

        head = machine.heads[head_index]
        tool_number = head.tool_number

        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.select_tool(tool_number), key="select-head"
        )

    def _progress_handler(self, sender, metrics):
        """Signal handler for job progress updates."""
        logger.debug(f"JobMonitor progress: {metrics}")
        if self._on_progress_callback:
            idle_add(self._on_progress_callback, metrics)

    async def _execute_monitored_job(
        self,
        ops: "Ops",
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Internal helper to execute an Ops object on a driver while managing
        a JobMonitor for progress reporting.
        """
        if self._current_monitor:
            msg = "Tried to start a job while another is running."
            logger.warning(msg)
            # A running job is a failure condition for starting a new one.
            raise RuntimeError(msg)

        if ops.is_empty():
            logger.warning("Job has no operations. Skipping execution.")
            if machine.driver:
                machine.driver.job_finished.send(machine.driver)
            return

        # Store the callback and create the monitor
        self._on_progress_callback = on_progress
        self._current_monitor = JobMonitor(ops)

        if self._on_progress_callback:
            logger.debug("Connecting progress handler to JobMonitor")
            self._current_monitor.progress_updated.connect(
                self._progress_handler
            )

        def cleanup_monitor(sender, **kwargs):
            """Cleans up the monitor when the job is done."""
            logger.debug("Job finished, cleaning up monitor.")
            if self._current_monitor:
                self._current_monitor.progress_updated.disconnect(
                    self._progress_handler
                )
                self._current_monitor = None
            self._on_progress_callback = None
            # Disconnect self to avoid being called again for this job
            machine.job_finished.disconnect(cleanup_monitor)

        # Connect to the machine's job_finished signal for cleanup
        machine.job_finished.connect(cleanup_monitor)

        # Signal that the job has started.
        idle_add(self.job_started.send, self)

        try:
            if machine.reports_granular_progress:
                await machine.driver.run(
                    ops,
                    machine,
                    self._editor.doc,
                    on_command_done=self._current_monitor.update_progress,
                )
            else:
                await machine.driver.run(
                    ops, machine, self._editor.doc, on_command_done=None
                )
                if self._current_monitor:
                    self._current_monitor.mark_as_complete()
        except Exception:
            # If run() throws an exception, the job_finished signal might not
            # be sent by the driver. We must clean up here.
            cleanup_monitor(machine)
            raise

    async def _run_frame_action(
        self,
        handle: JobArtifactHandle,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]],
    ):
        """The specific machine action for a framing job."""
        artifact = get_context().artifact_store.get(handle)
        if not isinstance(artifact, JobArtifact):
            raise ValueError("_run_frame_action received a non-JobArtifact")
        ops = artifact.ops

        head = machine.get_default_head()
        if not head.frame_power:
            logger.warning("Framing cancelled: Frame power is zero.")
            return  # This is a successful cancellation, not an error

        normalized_power = head.frame_power / head.max_power
        frame = ops.get_frame(
            power=normalized_power,
            speed=machine.max_travel_speed,
        )
        from ..core.ops import Ops

        frame_with_laser = Ops()
        frame_with_laser.set_laser(head.uid)
        frame_with_laser += frame * 20

        await self._execute_monitored_job(
            frame_with_laser, machine, on_progress
        )

    async def _run_send_action(
        self,
        handle: JobArtifactHandle,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]],
    ):
        """The specific machine action for a send job."""
        artifact = get_context().artifact_store.get(handle)
        if not isinstance(artifact, JobArtifact):
            raise ValueError("_run_send_action received a non-JobArtifact")
        ops = artifact.ops
        await self._execute_monitored_job(ops, machine, on_progress)

    def _start_job(
        self,
        machine: "Machine",
        job_name: str,
        final_job_action: Callable[..., Coroutine],
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> asyncio.Future:
        """
        Generic, non-blocking job starter that orchestrates assembly
        and execution.
        """
        try:
            # Get the future and its loop from the current async context.
            caller_loop = asyncio.get_running_loop()
            outer_future = caller_loop.create_future()
        except RuntimeError:
            # Fallback for non-async contexts (e.g., UI thread). The new
            # future gets associated with the main thread's default loop.
            outer_future = asyncio.Future()
            caller_loop = outer_future.get_loop()

        async def _run_entire_job(ctx):
            # This inner future is for managing the await inside this coroutine
            job_future = asyncio.get_running_loop().create_future()

            def _on_assembly_done(
                handle: Optional[JobArtifactHandle], error: Optional[Exception]
            ):
                if error:
                    logger.error(
                        f"Failed to assemble job for {job_name}",
                        exc_info=error,
                    )
                    if not job_future.done():
                        job_future.set_exception(error)
                    self._editor.notification_requested.send(
                        self,
                        message=_(f"{job_name.capitalize()} failed: {error}"),
                    )
                    if handle:
                        get_context().artifact_store.release(handle)
                    return

                if not handle:
                    logger.warning(
                        f"{job_name.capitalize()} job has no operations."
                    )
                    if not job_future.done():
                        job_future.set_result(None)
                    return

                async def _run_job_with_cleanup(ctx):
                    try:
                        await final_job_action(handle, machine, on_progress)
                        if not job_future.done():
                            job_future.set_result(True)
                    except Exception as e:
                        logger.error(
                            f"Failed to execute {job_name} job", exc_info=True
                        )
                        if not job_future.done():
                            job_future.set_exception(e)
                        self._editor.notification_requested.send(
                            self,
                            message=_(f"{job_name.capitalize()} failed: {e}"),
                        )
                    finally:
                        get_context().artifact_store.release(handle)

                self._editor.task_manager.add_coroutine(_run_job_with_cleanup)

            self._editor.pipeline.generate_job_artifact(
                when_done=_on_assembly_done
            )

            # Wait for the internal job to finish and transfer the result
            # to the outer future in a thread-safe way.
            try:
                result = await job_future
                if not outer_future.done():
                    caller_loop.call_soon_threadsafe(
                        outer_future.set_result, result
                    )
            except Exception as e:
                if not outer_future.done():
                    caller_loop.call_soon_threadsafe(
                        outer_future.set_exception, e
                    )

        self._editor.task_manager.add_coroutine(_run_entire_job)
        return outer_future

    def frame_job(
        self,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> asyncio.Future:
        """
        Asynchronously generates ops and runs a framing job.
        This is a non-blocking call that returns a future for completion.
        """
        return self._start_job(
            machine,
            job_name="framing",
            final_job_action=self._run_frame_action,
            on_progress=on_progress,
        )

    def send_job(
        self,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> asyncio.Future:
        """
        Asynchronously generates ops and sends the job to the machine.
        This is a non-blocking call that returns a future for completion.
        """
        return self._start_job(
            machine,
            job_name="sending",
            final_job_action=self._run_send_action,
            on_progress=on_progress,
        )

    def set_hold(self, machine: "Machine", is_requesting_hold: bool):
        """
        Adds a task to set the machine's hold state (pause/resume).
        """
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.set_hold(is_requesting_hold), key="set-hold"
        )

    def cancel_job(self, machine: "Machine"):
        """Adds a task to cancel the currently running job on the machine."""
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.cancel(), key="cancel-job"
        )

    def clear_alarm(self, machine: "Machine"):
        """Adds a task to clear any active alarm on the machine."""
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.clear_alarm(), key="clear-alarm"
        )

    def jog(self, machine: "Machine", axis: Axis, distance: float, speed: int):
        """
        Adds a task to jog the machine along a specific axis
        or combination of axes.
        """
        self._editor.task_manager.add_coroutine(
            lambda ctx: machine.jog(axis, distance, speed)
        )

    def home(self, machine: "Machine", axis: Optional[Axis] = None):
        """Adds a task to home a specific axis."""
        self._editor.task_manager.add_coroutine(
            lambda ctx: machine.driver.home(axis)
        )
