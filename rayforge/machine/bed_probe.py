"""Run a bed-mesh probing cycle over a grid of points.

This module is UI-free so the routine can be exercised against the
DummyDriver (or a fake) from backend tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.ops.axis import Axis

from .driver.driver import DeviceStatus

if TYPE_CHECKING:
    from .driver.driver import Driver, Pos

logger = logging.getLogger(__name__)

#: The GRBL serial driver gives up on any command after about 10 s
#: (raygeo's ``command_timeout``) and then reports a probe as not
#: triggered, while the controller keeps probing. A probe move is split
#: so that each part finishes well inside that time.
PROBE_COMMAND_BUDGET_S = 7.0

# GRBL's ``move_to`` is sent as a ``$J=`` jog. The controller answers
# "ok" as soon as the jog is queued, not when it finishes, and rejects
# ``G38.2`` while a jog runs (error 9). Wait for Idle after every move.
MOVE_SETTLE_S = 0.6
IDLE_POLL_S = 0.1
IDLE_TIMEOUT_S = 120.0

# A failed ``G38.2`` puts GRBL/FluidNC into a probe alarm. The motion
# ended at the probe target, so the position is still known and the
# alarm can be cleared without homing.
ALARM_APPEAR_TIMEOUT_S = 1.5
ALARM_CLEAR_TIMEOUT_S = 5.0


class BedProbeError(RuntimeError):
    """A probe point failed to trigger."""


class BedProbeAborted(RuntimeError):
    """The probe run was cancelled before completing."""


async def probe_bed_mesh(
    driver: Driver,
    *,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    feed_rate_mm_min: int,
    max_travel_mm: float,
    safe_z_mm: float,
    on_progress: Callable[[int, int, float, float, float], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> list[list[float]]:
    """Probe the bed at every grid point, in serpentine order.

    For each point the head rapids to ``(x, y, safe_z)``, probes down
    along Z by at most *max_travel_mm* at *feed_rate_mm_min*, records
    the trigger Z, and retracts straight up to *safe_z_mm* before
    traveling on. Serpentine ordering (alternating row direction)
    minimizes travel between consecutive points.

    :returns: ``ny`` rows of ``nx`` probed Z values (machine
        coordinates).
    :raises BedProbeError: a probe did not trigger, or a move failed.
    :raises BedProbeAborted: *should_abort* returned True between
        points (the head is retracted to safe Z first).
    """
    if nx < 2 or ny < 2:
        raise BedProbeError(_("mesh grid must be at least 2x2"))
    if max_travel_mm <= 0:
        raise BedProbeError(_("probe travel must be positive"))

    heights: list[list[float]] = []
    done = 0
    total = nx * ny
    # Where the head currently sits (the last commanded position);
    # retraction always lifts straight up from here, never travels.
    cur_x, cur_y = x0, y0
    for j in range(ny):
        row_indices = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        row: list[float] = [0.0] * nx
        for i in row_indices:
            if should_abort is not None and should_abort():
                await _retract(driver, cur_x, cur_y, safe_z_mm)
                raise BedProbeAborted
            x = x0 + i * dx
            y = y0 + j * dy
            try:
                # The first point includes Z because the head's
                # starting position is unknown. Every later point is
                # approached from safe Z, so the travel keeps Z and
                # moves in X/Y only.
                await _move(driver, x, y, safe_z_mm if done == 0 else None)
                # G38.2 targets an absolute Z in the active work
                # coordinates. Aim max_travel_mm below safe Z so the
                # probe moves down even when safe Z is negative.
                pos = await _probe_down(
                    driver,
                    safe_z_mm,
                    safe_z_mm - max_travel_mm,
                    feed_rate_mm_min,
                )
            except BedProbeAborted:
                raise
            except Exception as exc:
                await _retract(driver, x, y, safe_z_mm)
                raise BedProbeError(
                    _("probe failed at {x:.1f}, {y:.1f}: {error}").format(
                        x=x, y=y, error=exc
                    )
                ) from exc
            if pos is None or pos[2] is None:
                await _retract(driver, x, y, safe_z_mm)
                raise BedProbeError(
                    _("probe did not trigger at {x:.1f}, {y:.1f}").format(
                        x=x, y=y
                    )
                )
            z = pos[2]
            row[i] = z
            cur_x, cur_y = x, y
            # Lift straight up from the touched point before any
            # lateral travel, so a contact probe cannot drag along
            # the bed.
            await _move(driver, x, y, safe_z_mm)
            done += 1
            if on_progress is not None:
                on_progress(done, total, x, y, z)
        heights.append(row)

    return heights


async def _probe_down(
    driver: Driver,
    start_z: float,
    target_z: float,
    feed_rate_mm_min: int,
) -> Pos | None:
    """Probe down from work Z *start_z* to *target_z*.

    The move is split into parts short enough to finish within the
    driver's command timeout. A part that ends without contact leaves a
    probe alarm, which is cleared before the next part.

    :returns: the trigger position, or None if the probe reached
        *target_z* without contact.
    """
    step = max(0.05, feed_rate_mm_min / 60.0 * PROBE_COMMAND_BUDGET_S)
    z = start_z
    while True:
        part_target = max(target_z, z - step)
        pos = await driver.run_probe_cycle(
            Axis.Z, part_target, feed_rate_mm_min
        )
        if pos is not None and pos[2] is not None:
            return pos
        await _clear_probe_alarm(driver)
        if part_target <= target_z:
            return None
        z = part_target


async def _clear_probe_alarm(driver: Driver) -> None:
    """Clear the probe alarm a failed probe leaves behind.

    Waits briefly for the alarm to appear in the status reports, since
    a missed probe always raises one.
    """
    if getattr(driver, "state", None) is None:
        return
    deadline = time.monotonic() + ALARM_APPEAR_TIMEOUT_S
    while (
        driver.state.status != DeviceStatus.ALARM
        and time.monotonic() < deadline
    ):
        await asyncio.sleep(IDLE_POLL_S)
    if driver.state.status != DeviceStatus.ALARM:
        return
    await driver.clear_alarm()
    deadline = time.monotonic() + ALARM_CLEAR_TIMEOUT_S
    while driver.state.status == DeviceStatus.ALARM:
        if time.monotonic() > deadline:
            raise BedProbeError(_("could not clear the probe alarm"))
        await asyncio.sleep(IDLE_POLL_S)


async def _wait_idle(driver: Driver) -> None:
    """Wait for a queued jog to finish, so the next command can run."""
    if getattr(driver, "state", None) is None:
        return
    await asyncio.sleep(MOVE_SETTLE_S)
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    while True:
        status = driver.state.status
        if status == DeviceStatus.IDLE:
            return
        if status == DeviceStatus.ALARM:
            raise BedProbeError(_("the machine went into alarm"))
        if time.monotonic() > deadline:
            raise BedProbeError(
                _("timed out waiting for the machine to stop moving")
            )
        await asyncio.sleep(IDLE_POLL_S)


async def _move(
    driver: Driver,
    x: float,
    y: float,
    z: float | None,
    speed: float | None = None,
) -> None:
    await driver.move_to(x, y, z, speed)
    await _wait_idle(driver)


async def _retract(
    driver: Driver,
    x: float,
    y: float,
    z: float,
    speed: float | None = None,
):
    """Lift the head to *z* above the last probed position.

    Retraction must never mask a probing error, so failures are
    logged and swallowed here; the original error (or the abort)
    is what the caller needs to see.
    """
    try:
        await _move(driver, x, y, z, speed)
    except Exception:
        logger.warning("failed to retract to safe Z", exc_info=True)
