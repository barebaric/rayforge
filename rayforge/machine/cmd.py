from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..pipeline.job import generate_job_ops
from ..shared.tasker import task_mgr

if TYPE_CHECKING:
    from .models.machine import Machine
    from ..shared.tasker.context import ExecutionContext
    from ..doceditor.editor import DocEditor


logger = logging.getLogger(__name__)


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def home_machine(self, machine: "Machine"):
        """Adds a 'home' task to the task manager for the given machine."""
        driver = machine.driver
        task_mgr.add_coroutine(lambda ctx: driver.home(), key="home-machine")

    def frame_job(self, machine: "Machine"):
        """
        Generates ops for the current document and runs a framing job on the
        machine. This is an async operation managed by the task manager.
        """

        async def frame_coro(context: "ExecutionContext"):
            try:
                head = machine.heads[0]
                if not head.frame_power:
                    logger.warning("Framing cancelled: Frame power is zero.")
                    return

                ops = await generate_job_ops(
                    self._editor.doc,
                    machine,
                    self._editor.ops_generator,
                    context,
                )
                frame = ops.get_frame(
                    power=head.frame_power,
                    speed=machine.max_travel_speed,
                )
                # The frame op is a single pass; repeat it for visibility.
                frame *= 20
                await machine.driver.run(frame, machine, self._editor.doc)
            except Exception:
                logger.error("Failed to execute framing job", exc_info=True)
                raise

        task_mgr.add_coroutine(frame_coro, key="frame-job")

    def send_job(self, machine: "Machine"):
        """
        Generates ops for the current document and sends the job to the
        machine.
        This is an async operation managed by the task manager.
        """

        async def send_coro(context: "ExecutionContext"):
            try:
                ops = await generate_job_ops(
                    self._editor.doc,
                    machine,
                    self._editor.ops_generator,
                    context,
                )
                await machine.driver.run(ops, machine, self._editor.doc)
            except Exception:
                logger.error("Failed to send job to machine", exc_info=True)
                raise

        task_mgr.add_coroutine(send_coro, key="send-job")

    def set_hold(self, machine: "Machine", is_requesting_hold: bool):
        """
        Adds a task to set the machine's hold state (pause/resume).
        """
        driver = machine.driver
        task_mgr.add_coroutine(
            lambda ctx: driver.set_hold(is_requesting_hold), key="set-hold"
        )

    def cancel_job(self, machine: "Machine"):
        """Adds a task to cancel the currently running job on the machine."""
        driver = machine.driver
        task_mgr.add_coroutine(lambda ctx: driver.cancel(), key="cancel-job")

    def clear_alarm(self, machine: "Machine"):
        """Adds a task to clear any active alarm on the machine."""
        driver = machine.driver
        task_mgr.add_coroutine(
            lambda ctx: driver.clear_alarm(), key="clear-alarm"
        )
