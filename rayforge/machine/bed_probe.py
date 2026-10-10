"""Run a bed-mesh probing cycle over a grid of points.

This module is UI-free so the routine can be exercised against the
DummyDriver (or a fake) from backend tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.ops.axis import Axis

if TYPE_CHECKING:
    from .driver.driver import Driver

logger = logging.getLogger(__name__)


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
    the trigger Z, and retracts to *safe_z_mm* before traveling on.
    Serpentine ordering (alternating row direction) minimizes travel
    between consecutive points.

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
                await driver.move_to(x, y, safe_z_mm)
                pos = await driver.run_probe_cycle(
                    Axis.Z, -max_travel_mm, feed_rate_mm_min
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
            done += 1
            if on_progress is not None:
                on_progress(done, total, x, y, z)
        heights.append(row)

    await _retract(driver, cur_x, cur_y, safe_z_mm)
    return heights


async def _retract(driver: Driver, x: float, y: float, z: float):
    """Lift the head to *z* above the last probed position.

    Retraction must never mask a probing error, so failures are
    logged and swallowed here; the original error (or the abort)
    is what the caller needs to see.
    """
    try:
        await driver.move_to(x, y, z)
    except Exception:
        logger.warning("failed to retract to safe Z", exc_info=True)
