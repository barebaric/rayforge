from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Callable, Coroutine
from blinker import Signal
from ..pipeline.artifact import JobArtifact, JobArtifactHandle
from ..context import get_context
from .job_monitor import JobMonitor
from ..pipeline.encoder.context import GcodeContext, JobInfo
from ..shared.util.template import TemplateFormatter

if TYPE_CHECKING:
    from .models.machine import Machine
    from .driver.driver import Axis
    from ..doceditor.editor import DocEditor
    from ..core.ops import Ops
    from .models.laser import Laser


logger = logging.getLogger(__name__)


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self._scheduler = editor.task_manager.schedule_on_main_thread
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
            self._scheduler(self._on_progress_callback, metrics)

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

        def cleanup_monitor():
            """Cleans up the monitor when the job is done."""
            logger.debug("Job finished, cleaning up monitor.")
            if self._current_monitor:
                self._current_monitor.progress_updated.disconnect(
                    self._progress_handler
                )
                self._current_monitor = None
            self._on_progress_callback = None

        # Signal that the job has started.
        self._scheduler(self.job_started.send, self)

        try:
            if machine.reports_granular_progress:
                await machine.driver.run(
                    ops,
                    self._editor.doc,
                    on_command_done=self._current_monitor.update_progress,
                )
            else:
                await machine.driver.run(
                    ops, self._editor.doc, on_command_done=None
                )
                if self._current_monitor:
                    self._current_monitor.mark_as_complete()

            estimated_seconds = ops.estimate_time(
                default_cut_speed=machine.max_cut_speed,
                default_travel_speed=machine.max_travel_speed,
                acceleration=machine.acceleration,
            )
            estimated_hours = estimated_seconds / 3600.0
            machine.add_machine_hours(estimated_hours)
            logger.info(
                f"Job completed. Estimated time: {estimated_hours:.3f}h "
                f"added to machine hours."
            )
        finally:
            cleanup_monitor()

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
        if not head.frame_power_percent:
            logger.warning("Framing cancelled: Frame power is zero.")
            return  # This is a successful cancellation, not an error

        frame = ops.get_frame(
            power=head.frame_power_percent,
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

    async def _start_job(
        self,
        machine: "Machine",
        job_name: str,
        final_job_action: Callable[..., Coroutine],
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Generic, awaitable job starter that orchestrates assembly and
        execution.
        """
        handle: Optional[JobArtifactHandle] = None
        try:
            # 1. Await the job artifact generation from the pipeline
            handle = await self._editor.pipeline.generate_job_artifact_async()

            if not handle:
                logger.warning(
                    f"{job_name.capitalize()} job has no operations."
                )
                return

            # 2. Await the machine action (send/frame)
            await final_job_action(handle, machine, on_progress)

        except Exception as e:
            logger.error(
                f"Failed to assemble or execute {job_name} job", exc_info=True
            )
            self._editor.notification_requested.send(
                self, message=_(f"{job_name.capitalize()} failed: {e}")
            )
            # Re-raise the exception so the awaiting caller fails.
            raise
        finally:
            # 3. Always release the artifact to prevent memory leaks
            if handle:
                get_context().artifact_store.release(handle)

    async def frame_job(
        self,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Asynchronously generates ops and runs a framing job.
        This is an awaitable coroutine.
        """
        await self._start_job(
            machine,
            job_name="framing",
            final_job_action=self._run_frame_action,
            on_progress=on_progress,
        )

    async def send_job(
        self,
        machine: "Machine",
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Asynchronously generates ops and sends the job to the machine.
        This is an awaitable coroutine.
        """
        await self._start_job(
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

    def execute_macro_by_uid(self, machine: "Machine", macro_uid: str):
        """Finds a macro by UID, expands it, and runs it on the machine."""
        macro = machine.macros.get(macro_uid)
        if not macro or not macro.enabled:
            logger.warning(
                f"Macro with UID {macro_uid} not found or disabled."
            )
            return

        # A macro executed outside a job context has limited information.
        # We provide a dummy JobInfo for variables that might expect it.
        context = GcodeContext(
            machine=machine,
            doc=self._editor.doc,
            job=JobInfo(extents=(0, 0, 0, 0)),
        )
        formatter = TemplateFormatter(machine, context)
        expanded_lines = formatter.expand_macro(macro)
        gcode_to_run = "\n".join(expanded_lines)

        # We use the machine's run_raw method, which is simpler than building a
        # full job and allows macros to be self-contained.
        self._editor.task_manager.add_coroutine(
            lambda ctx: machine.run_raw(gcode_to_run),
            key=f"macro-{macro_uid}",
        )

    def set_power(self, head: "Laser", percent: float):
        """
        Adds a task to set the laser power to a specific percentage.

        Args:
            head: The laser head to control
            percent: Power percentage (0-1.0). 0 disables power.
        """
        config = get_context().config
        machine = config.machine
        if machine:
            self._editor.task_manager.add_coroutine(
                lambda ctx: machine.set_power(head, percent)
            )

    def home(self, machine: "Machine", axis: Optional[Axis] = None):
        """Adds a task to home a specific axis."""
        self._editor.task_manager.add_coroutine(lambda ctx: machine.home(axis))
