from __future__ import annotations
import logging
import asyncio
from typing import TYPE_CHECKING, Optional, Callable, Coroutine
from ..pipeline.artifact import ArtifactStore, JobArtifact, JobArtifactHandle

if TYPE_CHECKING:
    from .models.machine import Machine
    from .driver.driver import Axis
    from ..doceditor.editor import DocEditor


logger = logging.getLogger(__name__)


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

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

    # --- Refactored Job Execution Logic ---

    async def _run_frame_action(
        self, handle: JobArtifactHandle, machine: "Machine"
    ):
        """The specific machine action for a framing job."""
        artifact = ArtifactStore.get(handle)
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

        await machine.driver.run(frame_with_laser, machine, self._editor.doc)

    async def _run_send_action(
        self, handle: JobArtifactHandle, machine: "Machine"
    ):
        """The specific machine action for a send job."""
        artifact = ArtifactStore.get(handle)
        if not isinstance(artifact, JobArtifact):
            raise ValueError("_run_frame_action received a non-JobArtifact")
        ops = artifact.ops
        await machine.driver.run(ops, machine, self._editor.doc)

    def _start_job(
        self,
        machine: "Machine",
        job_name: str,
        final_job_action: Callable[[JobArtifactHandle, "Machine"], Coroutine],
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
                        ArtifactStore.release(handle)
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
                        await final_job_action(handle, machine)
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
                        ArtifactStore.release(handle)

                self._editor.task_manager.add_coroutine(_run_job_with_cleanup)

            self._editor.file.assemble_job_in_background(
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

    def frame_job(self, machine: "Machine") -> asyncio.Future:
        """
        Asynchronously generates ops and runs a framing job.
        This is a non-blocking call that returns a future for completion.
        """
        return self._start_job(
            machine,
            job_name="framing",
            final_job_action=self._run_frame_action,
        )

    def send_job(self, machine: "Machine") -> asyncio.Future:
        """
        Asynchronously generates ops and sends the job to the machine.
        This is a non-blocking call that returns a future for completion.
        """
        return self._start_job(
            machine, job_name="sending", final_job_action=self._run_send_action
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
