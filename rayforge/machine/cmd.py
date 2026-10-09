from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from gettext import gettext as _
from typing import TYPE_CHECKING

import numpy as np
from blinker import Signal
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandType

from ..context import get_context
from ..core.job_origin import StartFrom
from ..pipeline.artifact import JobArtifact
from ..pipeline.artifact.handle import BaseArtifactHandle
from ..pipeline.encoder.base import EncodedOutput
from ..pipeline.encoder.context import GcodeContext, JobInfo
from ..shared.util.template import TemplateFormatter
from .driver.driver import DeviceStatus, FrameCorner
from .job_monitor import JobMonitor
from .job_placement import JobPlacementError
from .kinematic_mapping import KinematicMapping
from .models.coordspace import MachineSpace
from .sanity import CheckMode, IssueSeverity, SanityChecker

if TYPE_CHECKING:
    from ..core.layer import Layer
    from ..doceditor.editor import DocEditor
    from .models.laser import Laser
    from .models.machine import Machine
    from .models.rotary_module import RotaryModule


logger = logging.getLogger(__name__)

# How long to wait for the machine to become idle after a framing or a
# job before the head is moved back to the start position.
RETURN_TO_START_TIMEOUT_S = 60.0
_IDLE_POLL_INTERVAL_S = 0.1


class JobAlreadyRunningError(RuntimeError):
    """
    Raised when a job is started while another one is still running.

    The message is user-facing (it surfaces via notification_requested)
    and tells the user how to resolve the situation instead of just
    describing the programming error.
    """


class MachineCmd:
    """Handles commands sent to the machine driver."""

    def __init__(self, editor: DocEditor):
        self._editor = editor
        self._scheduler = editor.task_manager.schedule_on_main_thread
        self.job_started = Signal()
        self._current_monitor: JobMonitor | None = None
        self._on_progress_callback: Callable[[dict], None] | None = None
        self._cancel_requested = False

    @property
    def is_job_running(self) -> bool:
        """Returns True if a monitored job is currently running."""
        return self._current_monitor is not None

    @property
    def current_monitor(self) -> JobMonitor | None:
        """Returns the monitor of the running job, if any."""
        return self._current_monitor

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
        execute: Callable[
            [Callable[[int], None | Awaitable[None]] | None],
            Coroutine,
        ]
        | None = None,
    ):
        """
        Internal helper to execute a job on a driver while managing
        a JobMonitor for progress reporting.

        The job runs the encoded pipeline output, unless *execute* is
        given: then it calls ``execute(on_command_done)`` instead, which
        runs a driver-defined action (framing) under the same monitoring.
        """
        if self._current_monitor:
            msg = _(
                "A job is already running. Wait for it to finish or "
                "press Stop to cancel it."
            )
            logger.warning(msg)
            # A running job is a failure condition for starting a new one.
            raise JobAlreadyRunningError(msg)

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
            estimated_seconds = ops.estimate_time(
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
                acceleration=machine.acceleration,
            )
            logger.info(
                f"JobMonitor: total_distance={ops.distance():.1f}mm, "
                f"estimated_seconds={estimated_seconds:.1f}s"
            )
            self._current_monitor = JobMonitor(
                ops,
                estimated_seconds=estimated_seconds,
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
                acceleration=machine.acceleration,
            )

            if self._on_progress_callback:
                logger.debug("Connecting progress handler to JobMonitor")
                self._current_monitor.progress_updated.connect(
                    self._progress_handler
                )

            # Signal that the job has started.
            self._scheduler(self.job_started.send, self)

            if execute is not None:
                on_command_done = (
                    self._current_monitor.update_progress
                    if (
                        machine.reports_granular_progress
                        and self._current_monitor
                    )
                    else None
                )
                await execute(on_command_done)
                if self._current_monitor and not (
                    machine.reports_granular_progress
                ):
                    self._current_monitor.mark_as_complete()
            elif encoded is None:
                # No driver-defined action: the pipeline must have
                # produced encoded output.
                raise RuntimeError("Pipeline did not produce encoded output.")
            elif machine.reports_granular_progress:
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
        """The specific machine action for a framing job.

        Builds the frame outline in command space — through the same
        transforms a job gets — and hands the corners to the driver,
        which frames in whatever way fits the controller (see
        Driver.frame()).
        """
        if not isinstance(artifact, JobArtifact):
            raise TypeError("_run_frame_action received a non-JobArtifact")

        head = machine.get_default_laser_head()
        if head is None:
            raise ValueError("Machine has no laser heads configured.")

        frame_speed = (
            head.frame_speed
            if head.frame_speed > 0
            else machine.max_travel_speed
        )
        repeat_count = max(1, head.frame_repeat_count)

        # A moves-only trace carrying the outline through the same
        # transforms a job gets.
        frame_ops = _build_frame_trace(artifact.ops.rect(), repeat_count)

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

        # Transform world-space frame ops to machine space, then to
        # command space: the emitted coordinates must be relative to
        # the active WCS origin because the controller adds the WCS
        # offset back (issue #362). The regular send path performs the
        # same adjustment in the machine-transform stage. While
        # pointer alignment is on, the command WCS offset includes the
        # pointer offset, so the frame outline is traced by the
        # pointer dot rather than the cutting beam.
        space = MachineSpace.from_machine(machine)
        combined = space.get_world_to_machine_matrix()
        if machine.reverse_z_axis:
            z_flip = np.eye(4)
            z_flip[2, 2] = -1.0
            combined = z_flip @ combined
        frame_ops.transform(combined)

        _warn_if_frame_exceeds_travel(self._editor, machine, frame_ops.rect())

        to_command = space.get_machine_to_command_matrix(
            wcs_offset=machine.get_command_wcs_offset(),
            wcs_is_workarea_origin=machine.wcs_origin_is_workarea_origin,
        )
        frame_ops.transform(to_command)

        # AXIS_REPLACEMENT modules encode the rotary degrees into the
        # replaced machine axis after the world→machine transform.
        if rotary_module is not None and rotary_module.is_replacement():
            KinematicMapping.degrees_to_mm_pass(
                frame_ops,
                rotary_module.mm_per_rotation,
                target_axis=rotary_module.axis,
            )

        corners = _frame_trace_corners(
            frame_ops, len(_frame_rect_corners(artifact.ops.rect()))
        )

        async def execute(
            on_command_done: Callable[[int], None | Awaitable[None]] | None,
        ):
            await machine.driver.frame(
                corners,
                frame_speed,
                self._editor.doc,
                repeat_count=repeat_count,
                corner_pause_s=head.frame_corner_pause,
                power_fraction=head.frame_power_percent,
                on_command_done=on_command_done,
            )

        await self._execute_monitored_job(
            frame_ops,
            machine,
            on_progress=on_progress,
            execute=execute,
        )

    async def _run_send_action(
        self,
        artifact: JobArtifact,
        machine: Machine,
        on_progress: Callable[[dict], None] | None,
    ):
        """The specific machine action for a send job.

        While a pointer dry-run was requested, the pipeline encoded
        this artifact with all laser power capped at the head's
        framing power and the pointer offset folded into the WCS
        offsets, so the pointer dot traces an unburnable toolpath.
        """
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
        pointer_dry_run: bool = False,
    ):
        """
        Generic, awaitable job executor that orchestrates artifact
        assembly and execution.
        """
        handle: BaseArtifactHandle | None = None
        artifact_store = self._editor.pipeline.artifact_store
        pipeline = self._editor.pipeline
        shift_used = False
        self._cancel_requested = False
        return_point = self._start_position_in_command_space(machine)

        try:
            if pointer_dry_run and machine.has_pointer_offset():
                # Generate the job once with the pointer offset folded
                # into the WCS offsets so the pointer dot traces the
                # toolpath, and with all laser power capped at the
                # head's framing power (via get_job_power_cap in the
                # encode context) so the trace cannot burn. The flags
                # are only visible during generation; the shifted
                # artifact is discarded afterwards so the next regular
                # send burns unshifted at job power.
                machine.pointer_job_shift_enabled = True
                shift_used = True
                pipeline.invalidate_job_output()
            try:
                # 1. Await the job artifact generation from the pipeline
                handle = await pipeline.generate_job_artifact_async()
            finally:
                machine.pointer_job_shift_enabled = False

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
                if shift_used:
                    # The cached output is shifted; drop it while our
                    # checkout keeps the artifact alive.
                    pipeline.invalidate_job_output()

                if isinstance(artifact, JobArtifact):
                    self._check_job_placement(artifact, machine)
                await final_job_action(artifact, machine, on_progress)

            if return_point is not None and not self._cancel_requested:
                await self._return_to_start(machine, return_point)

        except JobAlreadyRunningError:
            # Already logged as a warning by the guard; not an error.
            raise
        except Exception:
            logger.exception("Failed to assemble or execute job")
            # Manually release handle on error if checkout was not entered
            if handle and "artifact" not in locals():
                artifact_store.release(handle)
            raise

    def _start_position_in_command_space(
        self, machine: Machine
    ) -> tuple[float, float] | None:
        """
        The head position before a Start From "Current Position" job,
        in command coordinates of the active WCS, or None in any other
        mode or when the position is unknown.
        """
        job_origin = self._editor.doc.job_origin
        if job_origin.start_from != StartFrom.CURRENT_POSITION:
            return None
        pos_x, pos_y = machine.device_state.machine_pos[:2]
        if pos_x is None or pos_y is None:
            return None
        off_x, off_y, _z = machine.panel.get_command_offset(
            wcs_offset=machine.get_active_wcs_offset(),
            wcs_is_workarea_origin=machine.wcs_origin_is_workarea_origin,
        )
        return (pos_x - off_x, pos_y - off_y)

    def _check_job_placement(self, artifact: JobArtifact, machine: Machine):
        """
        Refuse a job placed via Start From that leaves the machine
        travel or enters a no-go zone.

        The UI runs the same checks before sending and lets the user
        override them; this is the backend guard for jobs whose
        position depends on where the head or the WCS zero is, which
        the user cannot see on the canvas.

        Raises:
            JobPlacementError: With the issues found.
        """
        if self._editor.doc.job_origin.is_absolute:
            return
        report = SanityChecker(machine).check(
            artifact.ops, mode=CheckMode.FAST
        )
        errors = [
            issue.message
            for issue in report.issues
            if issue.severity == IssueSeverity.ERROR
        ]
        if errors:
            raise JobPlacementError(
                _(
                    "The job does not fit at the chosen start position: "
                    "{issues}"
                ).format(issues="; ".join(errors))
            )

    async def _return_to_start(
        self, machine: Machine, point: tuple[float, float]
    ):
        """
        Move the head back to where a "Current Position" job or frame
        started, once the machine is idle again.

        The dialect postscript usually returns to X0 Y0; without this,
        Frame followed by Start would place the job at the WCS zero.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + RETURN_TO_START_TIMEOUT_S
        while machine.device_state.status != DeviceStatus.IDLE:
            if loop.time() >= deadline or self._cancel_requested:
                logger.warning(
                    "Machine did not become idle; head not returned to "
                    "the start position."
                )
                self._editor.notification_requested.send(
                    self,
                    message=_(
                        "The head was not moved back to the start "
                        "position. Check its position before the next "
                        "Frame or Start."
                    ),
                )
                return
            await asyncio.sleep(_IDLE_POLL_INTERVAL_S)
        await machine.driver.move_to(point[0], point[1])

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
        except JobAlreadyRunningError as e:
            # An expected refusal, not a failure: the guard's message
            # already tells the user what to do.
            self._editor.notification_requested.send(
                self,
                message=str(e),
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
        pointer_dry_run: bool = False,
    ):
        """
        Asynchronously generates ops and sends the job to the machine.
        This is an awaitable coroutine.

        With pointer_dry_run, the job is generated with the pointer
        offset folded into the WCS offsets, so the pointer dot traces
        the toolpath while the beam runs displaced by the offset, and
        all laser power is capped at the head's framing power so the
        trace does not burn the material.
        """
        try:
            await self._start_job(
                machine,
                final_job_action=self._run_send_action,
                on_progress=on_progress,
                pointer_dry_run=pointer_dry_run,
            )
        except JobAlreadyRunningError as e:
            # An expected refusal, not a failure: the guard's message
            # already tells the user what to do.
            self._editor.notification_requested.send(
                self,
                message=str(e),
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
        self._cancel_requested = True
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

    def move_to(
        self,
        machine: Machine,
        x: float,
        y: float,
        z: float | None = None,
        speed: float | None = None,
    ):
        """
        Adds a task to move to an absolute position.

        The speed is given in mm/min; when None the driver applies its
        default move speed.
        """
        driver = machine.driver
        if driver:
            self._editor.task_manager.add_coroutine(
                lambda ctx: driver.move_to(x, y, z, speed), key="move-to"
            )


def _warn_if_frame_exceeds_travel(
    editor: DocEditor,
    machine: Machine,
    frame_rect: tuple[float, float, float, float],
):
    """
    Warn when the pointer-shifted frame leaves the machine travel.

    While pointer alignment is on, the beam traces the frame one
    pointer offset behind the outline, which can leave the reachable
    area even though the outline itself is on the bed. The frame still
    runs; the warning explains why the trace may stop short of a side.
    """
    if not machine.pointer_alignment_enabled:
        return
    dx, dy = machine.get_pointer_offset()
    min_x, min_y, max_x, max_y = frame_rect
    width, height = machine.axis_extents
    x_min = -width if machine.reverse_x_axis else 0.0
    x_max = 0.0 if machine.reverse_x_axis else width
    y_min = -height if machine.reverse_y_axis else 0.0
    y_max = 0.0 if machine.reverse_y_axis else height
    corners = [
        (min_x - dx, min_y - dy),
        (min_x - dx, max_y - dy),
        (max_x - dx, max_y - dy),
        (max_x - dx, min_y - dy),
    ]
    if all(x_min <= x <= x_max and y_min <= y <= y_max for x, y in corners):
        return
    editor.notification_requested.send(
        editor,
        message=_(
            "Pointer alignment shifts the frame partly outside the "
            "machine travel; the trace may stop short of a side."
        ),
        persistent=True,
    )


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


def _frame_rect_corners(
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """The frame outline corners for a bounding *rect*.

    The list is closed: the first corner repeats last so a trace of
    consecutive segments returns to the start.
    """
    min_x, min_y, max_x, max_y = rect
    return [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y),
    ]


def _build_frame_trace(
    rect: tuple[float, float, float, float],
    repeat_count: int,
) -> Ops:
    """Build the frame outline as a moves-only trace.

    This carries the outline through the frame's coordinate transforms
    and feeds the job estimate; the driver builds the actual command
    stream from the traced corners (see ``Driver.frame``).
    """
    trace = Ops()
    corners = _frame_rect_corners(rect)
    for _repeat in range(max(1, repeat_count)):
        for corner in corners:
            trace.move_to(*corner)
    return trace


def _frame_trace_corners(
    ops: Ops,
    corner_count: int,
) -> list[FrameCorner]:
    """Read the first-pass outline corners back from the trace.

    Returns the position and extra axes of the first *corner_count*
    MOVE_TO commands — the transformed corners in command space, in
    trace order. The extra axes carry the rotary degrees of rotary
    frames.
    """
    corners = []
    for i in range(ops.len()):
        if ops.command_type(i) == CommandType.MOVE_TO:
            x, y, _z = ops.endpoint(i)
            corners.append((x, y, ops.extra_axes(i)))
            if len(corners) == corner_count:
                break
    return corners
