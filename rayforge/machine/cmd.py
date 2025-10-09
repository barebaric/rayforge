from __future__ import annotations
import logging
import asyncio
from typing import TYPE_CHECKING, Optional, Tuple

from ..pipeline.artifact.store import ArtifactStore
from ..pipeline.artifact.handle import ArtifactHandle

if TYPE_CHECKING:
    from .models.machine import Machine
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

    def frame_job(self, machine: "Machine") -> asyncio.Future:
        """
        Asynchronously generates ops and runs a framing job.
        This is a non-blocking call that returns a future for completion.
        """
        job_future = asyncio.get_running_loop().create_future()

        def _on_assembly_done(
            result: Optional[Tuple[float, Optional[ArtifactHandle]]],
            error: Optional[Exception],
        ):
            if error:
                logger.error(
                    "Failed to assemble job for framing", exc_info=error
                )
                if not job_future.done():
                    job_future.set_exception(error)
                self._editor.notification_requested.send(
                    self,
                    message=_("Framing failed: {error}").format(error=error),
                )
                return

            if not result:
                exc = ValueError("Assembly for framing returned no result.")
                logger.error(exc)
                if not job_future.done():
                    job_future.set_exception(exc)
                return

            _time, handle = result

            if not handle:
                logger.warning("Framing job has no operations to run.")
                if not job_future.done():
                    job_future.set_result(None)
                return

            # This coroutine will now run in the task manager
            async def _run_frame(ctx):
                try:
                    assert handle is not None, "Handle must exist to run frame"
                    artifact = ArtifactStore.get(handle)
                    ops = artifact.ops

                    head = machine.get_default_head()
                    if not head.frame_power:
                        logger.warning(
                            "Framing cancelled: Frame power is zero."
                        )
                        if not job_future.done():
                            job_future.set_result(None)
                        return

                    normalized_power = head.frame_power / head.max_power
                    frame = ops.get_frame(
                        power=normalized_power,
                        speed=machine.max_travel_speed,
                    )
                    from ..core.ops import Ops

                    frame_with_laser = Ops()
                    frame_with_laser.set_laser(head.uid)
                    frame_with_laser += frame * 20

                    await machine.driver.run(
                        frame_with_laser, machine, self._editor.doc
                    )
                    if not job_future.done():
                        job_future.set_result(True)
                except Exception as e:
                    logger.error(
                        "Failed to execute framing job", exc_info=True
                    )
                    if not job_future.done():
                        job_future.set_exception(e)
                    self._editor.notification_requested.send(
                        self,
                        message=_("Framing failed: {error}").format(error=e),
                    )
                finally:
                    if handle:
                        ArtifactStore.release(handle)

            self._editor.task_manager.add_coroutine(_run_frame)

        self._editor.file.assemble_job_in_background(
            when_done=_on_assembly_done
        )
        return job_future

    def send_job(self, machine: "Machine") -> asyncio.Future:
        """
        Asynchronously generates ops and sends the job to the machine.
        This is a non-blocking call that returns a future for completion.
        """
        job_future = asyncio.get_running_loop().create_future()

        def _on_assembly_done(
            result: Optional[Tuple[float, Optional[ArtifactHandle]]],
            error: Optional[Exception],
        ):
            if error:
                logger.error(
                    "Failed to assemble job for sending", exc_info=error
                )
                if not job_future.done():
                    job_future.set_exception(error)
                self._editor.notification_requested.send(
                    self, message=_("Job failed: {error}").format(error=error)
                )
                return

            if not result:
                exc = ValueError("Assembly for sending returned no result.")
                logger.error(exc)
                if not job_future.done():
                    job_future.set_exception(exc)
                return

            _time, handle = result

            if not handle:
                logger.warning("Job has no operations to run.")
                if not job_future.done():
                    job_future.set_result(None)
                return

            async def _run_job(ctx):
                try:
                    assert handle is not None, "Handle must exist to run job"
                    artifact = ArtifactStore.get(handle)
                    ops = artifact.ops
                    await machine.driver.run(ops, machine, self._editor.doc)
                    if not job_future.done():
                        job_future.set_result(True)
                except Exception as e:
                    logger.error(
                        "Failed to send job to machine", exc_info=True
                    )
                    if not job_future.done():
                        job_future.set_exception(e)
                    self._editor.notification_requested.send(
                        self, message=_("Job failed: {error}").format(error=e)
                    )
                finally:
                    if handle:
                        ArtifactStore.release(handle)

            self._editor.task_manager.add_coroutine(_run_job)

        self._editor.file.assemble_job_in_background(
            when_done=_on_assembly_done
        )
        return job_future

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
