from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from gettext import gettext as _
from typing import TYPE_CHECKING

import numpy as np
from blinker import Signal
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from ..context import get_context
from ..pipeline.artifact import JobArtifact
from ..pipeline.artifact.handle import BaseArtifactHandle
from ..pipeline.encoder.base import EncodedOutput
from ..pipeline.encoder.context import GcodeContext, JobInfo
from ..shared.util.template import TemplateFormatter
from .driver import get_driver_cls
from .driver.dummy import NoDeviceDriver
from .job_monitor import JobMonitor
from .kinematic_mapping import KinematicMapping
from .models.coordspace import MachineSpace

if TYPE_CHECKING:
    from ..core.layer import Layer
    from ..doceditor.editor import DocEditor
    from .models.laser import Laser
    from .models.machine import Machine
    from .models.rotary_module import RotaryModule


logger = logging.getLogger(__name__)


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: DocEditor):
        self._editor = editor
        self._scheduler = editor.task_manager.schedule_on_main_thread
        self.job_started = Signal()
        self._current_monitor: JobMonitor | None = None
        self._on_progress_callback: Callable[[dict], None] | None = None

    @property
    def is_job_running(self) -> bool:
        """Returns True if a monitored job is currently running."""
        return self._current_monitor is not None

    def select_tool(self, machine: Machine, head_index: int):
        """Adds a 'select_head' task to the task manager."""
        if not (0 <= head_index < len(machine.heads)):
            logger.error(f"Invalid head index {head_index} for tool selection")
            return

        head = machine.heads[head_index]
        tool_number = head.tool_number

        self._editor.task_manager.add_coroutine(
            lambda ctx: machine.select_tool(tool_number), key="select-head"
        )

    def _progress_handler(self, sender, metrics):
        """Signal handler for job progress updates."""
        logger.debug(f"JobMonitor progress: {metrics}")
        if self._on_progress_callback:
            self._scheduler(self._on_progress_callback, metrics)

    async def _execute_monitored_job(
        self,
        ops: Ops,
        machine: Machine,
        on_progress: Callable[[dict], None] | None = None,
        encoded: EncodedOutput | None = None,
    ):
        """
        Internal helper to execute a job on a driver while managing
        a JobMonitor for progress reporting.
        """
        if self._current_monitor:
            msg = _("Tried to start a job while another is running.")
            logger.warning(msg)
            # A running job is a failure condition for starting a new one.
            raise RuntimeError(msg)

        if ops.is_empty():
            logger.warning("Job has no operations. Skipping execution.")
            if machine.driver:
                machine.driver.job_finished.send(machine.driver)
            return

        # Store the callback
        self._on_progress_callback = on_progress

        def cleanup_monitor():
            """Cleans up the monitor when the job is done."""
            logger.debug("Job finished, cleaning up monitor.")
            if self._current_monitor:
                try:
                    self._current_monitor.progress_updated.disconnect(
                        self._progress_handler
                    )
                finally:
                    # Ensure the flag is cleared even if disconnect fails.
                    self._current_monitor = None
            self._on_progress_callback = None

        try:
            self._current_monitor = JobMonitor(ops)

            if self._on_progress_callback:
                logger.debug("Connecting progress handler to JobMonitor")
                self._current_monitor.progress_updated.connect(
                    self._progress_handler
                )

            # Signal that the job has started.
            self._scheduler(self.job_started.send, self)

            # Pipeline must have produced encoded output.
            if encoded is None:
                raise RuntimeError("Pipeline did not produce encoded output.")

            if machine.reports_granular_progress:
                await machine.driver.run(
                    encoded,
                    self._editor.doc,
                    ops,
                    on_command_done=self._current_monitor.update_progress,
                )
            else:
                await machine.driver.run(
                    encoded,
                    self._editor.doc,
                    ops,
                    on_command_done=None,
                )
                if self._current_monitor:
                    self._current_monitor.mark_as_complete()

            estimated_seconds = ops.estimate_time(
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
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
        artifact: JobArtifact,
        machine: Machine,
        on_progress: Callable[[dict], None] | None,
    ):
        """The specific machine action for a framing job."""
        if not isinstance(artifact, JobArtifact):
            raise TypeError("_run_frame_action received a non-JobArtifact")
        ops = artifact.ops

        head = machine.get_default_laser_head()
        if head is None:
            raise ValueError("Machine has no laser heads configured.")
        if not head.frame_power_percent:
            logger.warning("Framing cancelled: Frame power is zero.")
            return

        frame_speed = (
            head.frame_speed
            if head.frame_speed > 0
            else machine.max_travel_speed
        )

        min_x, min_y, max_x, max_y = ops.rect()

        frame_ops = Ops()
        frame_ops.set_head(head.uid)
        frame_ops.set_power(head.frame_power_percent)
        frame_ops.set_feed_rate(frame_speed)

        corners = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        ]
        prev = corners[0]
        for corner in corners[1:]:
            frame_ops.move_to(*prev)
            frame_ops.line_to(*corner)
            if head.frame_corner_pause > 0:
                frame_ops.dwell(head.frame_corner_pause * 1000)
            prev = corner

        # Apply the active layer's rotary mapping so the frame drives
        # the rotary axis instead of the physical Y axis (issue #356).
        # The regular send path applies this in the machine-transform
        # stage, but the frame ops are built on the fly and bypass the
        # pipeline, so map them here in world space before the
        # world→machine transform.
        rotary_module = _apply_frame_rotary_mapping(
            frame_ops,
            machine,
            self._editor.doc.active_layer,
        )

        frame_with_laser = frame_ops * head.frame_repeat_count
        frame_with_laser.job_end()

        # Transform world-space frame ops to machine space.
        space = MachineSpace.from_machine(machine)
        combined = space.get_world_to_machine_matrix()
        if machine.reverse_z_axis:
            z_flip = np.eye(4)
            z_flip[2, 2] = -1.0
            combined = z_flip @ combined
        frame_with_laser.transform(combined)

        # AXIS_REPLACEMENT modules encode the rotary degrees into the
        # replaced machine axis after the world→machine transform.
        if rotary_module is not None and rotary_module.is_replacement():
            KinematicMapping.degrees_to_mm_pass(
                frame_with_laser,
                rotary_module.mm_per_rotation,
                target_axis=rotary_module.axis,
            )

        # Encode via the driver encoder (no pre-processing needed).
        encoder = _create_driver_encoder(machine)
        encoded = encoder.encode(frame_with_laser, machine, self._editor.doc)

        await self._execute_monitored_job(
            frame_with_laser,
            machine,
            on_progress=on_progress,
            encoded=encoded,
        )

    async def _run_send_action(
        self,
        artifact: JobArtifact,
        machine: Machine,
        on_progress: Callable[[dict], None] | None,
    ):
        """The specific machine action for a send job."""
        if not isinstance(artifact, JobArtifact):
            raise TypeError("_run_send_action received a non-JobArtifact")

        await self._execute_monitored_job(
            artifact.ops,
            machine,
            on_progress=on_progress,
            encoded=artifact.encoded_output,
        )

    async def _start_job(
        self,
        machine: Machine,
        final_job_action: Callable[..., Coroutine],
        on_progress: Callable[[dict], None] | None = None,
    ):
        """
        Generic, awaitable job executor that orchestrates artifact
        assembly and execution.
        """
        handle: BaseArtifactHandle | None = None
        artifact_store = self._editor.pipeline.artifact_store

        try:
            # 1. Await the job artifact generation from the pipeline
            handle = await self._editor.pipeline.generate_job_artifact_async()

            if not handle:
                logger.warning("Job has no operations.")
                return

            # 2. Use the safe context manager to acquire and release the
            # artifact
            with artifact_store.checkout_handle(handle) as artifact:
                if not artifact:
                    raise ValueError(
                        "Failed to retrieve artifact from handle."
                    )

                await final_job_action(artifact, machine, on_progress)

        except Exception:
            logger.exception("Failed to assemble or execute job")
            # Manually release handle on error if checkout was not entered
            if handle and "artifact" not in locals():
                artifact_store.release(handle)
            raise

    async def frame_job(
        self,
        machine: Machine,
        on_progress: Callable[[dict], None] | None = None,
    ):
        """
        Asynchronously generates ops and runs a framing job.
        This is an awaitable coroutine.
        """
        try:
            await self._start_job(
                machine,
                final_job_action=self._run_frame_action,
                on_progress=on_progress,
            )
        except Exception as e:
            self._editor.notification_requested.send(
                self,
                message=_("Framing failed: {error}").format(error=e),
            )
            raise

    async def send_job(
        self,
        machine: Machine,
        on_progress: Callable[[dict], None] | None = None,
    ):
        """
        Asynchronously generates ops and sends the job to the machine.
        This is an awaitable coroutine.
        """
        try:
            await self._start_job(
                machine,
                final_job_action=self._run_send_action,
                on_progress=on_progress,
            )
        except Exception as e:
            self._editor.notification_requested.send(
                self,
                message=_("Sending failed: {error}").format(error=e),
            )
            raise

    def run_send_job(self, machine: Machine):
        """
        Schedules the send_job coroutine to run via the task manager.
        """
        self._editor.task_manager.add_coroutine(
            lambda ctx: self.send_job(machine),
            key="send-job",
        )

    def set_hold(self, machine: Machine, is_requesting_hold: bool):
        """
        Adds a task to set the machine's hold state (pause/resume).
        """
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.set_hold(is_requesting_hold), key="set-hold"
        )

    def cancel_job(self, machine: Machine):
        """Adds a task to cancel the currently running job on the machine."""
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.cancel(), key="cancel-job"
        )

    def clear_alarm(self, machine: Machine):
        """Adds a task to clear any active alarm on the machine."""
        driver = machine.driver
        self._editor.task_manager.add_coroutine(
            lambda ctx: driver.clear_alarm(), key="clear-alarm"
        )

    def jog(self, machine: Machine, deltas: dict[Axis, float], speed: int):
        """
        Adds a task to jog the machine along specific axes.
        """
        self._editor.task_manager.add_coroutine(
            lambda ctx: machine.jog(deltas, speed)
        )

    def execute_macro_by_uid(self, machine: Machine, macro_uid: str):
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

    def set_power(
        self,
        head: Laser,
        percent: float,
        machine: Machine | None = None,
    ):
        """
        Adds a task to set the laser power to a specific percentage.

        Args:
            head: The laser head to control
            percent: Power percentage (0-1.0). 0 disables power.
        """
        if machine is None:
            config = get_context().config
            machine = config.machine
        if machine:
            self._editor.task_manager.add_coroutine(
                lambda ctx: machine.set_power(head, percent)
            )

    def set_focus_power(
        self,
        head: Laser,
        percent: float,
        machine: Machine | None = None,
    ):
        """
        Adds a task to set the laser power for focus mode.

        Args:
            head: The laser head to control
            percent: Power percentage (0-1.0). 0 disables power.
        """
        if machine is None:
            config = get_context().config
            machine = config.machine
        if machine:
            self._editor.task_manager.add_coroutine(
                lambda ctx: machine.set_focus_power(head, percent)
            )

    def home(self, machine: Machine, axis: Axis | None = None):
        """Adds a task to home a specific axis."""
        self._editor.task_manager.add_coroutine(lambda ctx: machine.home(axis))

    def move_to(self, machine: Machine, x: float, y: float):
        """Adds a task to move to an absolute position."""
        driver = machine.driver
        if driver:
            self._editor.task_manager.add_coroutine(
                lambda ctx: driver.move_to(x, y), key="move-to"
            )


def _create_driver_encoder(machine: Machine):
    """Instantiate the machine's driver encoder."""
    if machine.driver_name:
        try:
            driver_cls = get_driver_cls(machine.driver_name)
        except (ValueError, ImportError):
            driver_cls = NoDeviceDriver
    else:
        driver_cls = NoDeviceDriver
    return driver_cls.create_encoder(machine)


def _apply_frame_rotary_mapping(
    frame_ops: Ops,
    machine: Machine,
    layer: Layer | None,
) -> RotaryModule | None:
    """Apply *layer*'s rotary mapping to the world-space frame ops.

    Converts the frame's Y movement into rotary degrees (stored in the
    rotary axis' ``extra_axes``) and pins Y to the cylinder axis
    position, so the frame sweeps the X axis while rotating the rotary
    instead of driving the physical Y axis (issue #356).

    Returns the resolved rotary module when a valid one applies, or
    ``None`` when the layer is flat or has no usable rotary module.  The
    caller uses the returned module to run the AXIS_REPLACEMENT
    degrees→machine-units pass after the world→machine transform.
    """
    if layer is None or not layer.rotary_enabled:
        return None
    module = machine.get_rotary_module_for_layer(layer)
    if module is None:
        return None
    # AXIS_REPLACEMENT modules can only map to a real machine axis when
    # they have a travel-per-rotation or target one of the XYZ axes.
    if module.is_replacement() and (
        module.mm_per_rotation <= 0
        and module.axis not in (Axis.X, Axis.Y, Axis.Z)
    ):
        return None
    mapping = KinematicMapping.from_rotary_module(
        module,
        layer.rotary_diameter,
        apply_gear_ratio=True,
    )
    if mapping is not None:
        mapping.apply(frame_ops)
    return module
