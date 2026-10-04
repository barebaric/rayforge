"""Tests for the bed-mesh probing routine."""

import math
from typing import cast

import pytest

from rayforge.machine.bed_probe import (
    BedProbeAborted,
    BedProbeError,
    probe_bed_mesh,
)
from rayforge.machine.driver.driver import Driver

SAFE_Z = 5.0


class FakeDriver:
    """Records moves and reports a synthetic wavy bed surface."""

    supports_probing = True

    def __init__(self, fail_at=None, surface=None):
        self.moves: list[tuple[float, float, float | None]] = []
        self.fail_at: tuple[float, float] | None = fail_at
        self.surface = surface or (
            lambda x, y: 0.5 * math.sin(x / 20.0) + 0.2 * math.cos(y / 15.0)
        )

    async def move_to(self, pos_x, pos_y, pos_z=None, speed=None):
        self.moves.append((pos_x, pos_y, pos_z))

    async def run_probe_cycle(self, axis, max_travel, feed_rate):
        x, y, _z = self.moves[-1]
        if self.fail_at == (x, y):
            return None
        return (x, y, self.surface(x, y))


def _points(result):
    return [z for row in result for z in row]


@pytest.mark.asyncio
async def test_probes_all_points_in_serpentine_order():
    driver = FakeDriver()
    result = await probe_bed_mesh(
        cast(Driver, driver),
        x0=0.0,
        y0=0.0,
        dx=10.0,
        dy=10.0,
        nx=3,
        ny=3,
        feed_rate_mm_min=100,
        max_travel_mm=10.0,
        safe_z_mm=SAFE_Z,
    )
    assert len(result) == 3
    assert all(len(row) == 3 for row in result)
    # Heights match the synthetic surface.
    for j, row in enumerate(result):
        for i, z in enumerate(row):
            assert z == pytest.approx(driver.surface(i * 10.0, j * 10.0))

    # Serpentine: row 0 runs x ascending, row 1 descending, row 2
    # ascending; every travel move happens at the safe Z height.
    probe_points = [
        (i * 10.0, j * 10.0)
        for j in range(3)
        for i in ([0, 1, 2] if j % 2 == 0 else [2, 1, 0])
    ]
    travel_points = [(m[0], m[1]) for m in driver.moves]
    assert travel_points[:9] == probe_points
    assert all(m[2] == SAFE_Z for m in driver.moves)


@pytest.mark.asyncio
async def test_retracts_to_safe_z_after_each_point():
    driver = FakeDriver()
    await probe_bed_mesh(
        cast(Driver, driver),
        x0=0.0,
        y0=0.0,
        dx=10.0,
        dy=10.0,
        nx=2,
        ny=2,
        feed_rate_mm_min=100,
        max_travel_mm=10.0,
        safe_z_mm=SAFE_Z,
    )
    # move_to is only ever called with the safe Z travel height; the
    # probe plunge/retract itself lives in run_probe_cycle.
    assert len(driver.moves) == 5  # 4 points + final retract
    assert all(m[2] == SAFE_Z for m in driver.moves)


@pytest.mark.asyncio
async def test_progress_callback_fires_per_point():
    driver = FakeDriver()
    seen = []

    def on_progress(done, total, x, y, z):
        seen.append((done, total, x, y))

    await probe_bed_mesh(
        cast(Driver, driver),
        x0=0.0,
        y0=0.0,
        dx=10.0,
        dy=10.0,
        nx=2,
        ny=2,
        feed_rate_mm_min=100,
        max_travel_mm=10.0,
        safe_z_mm=SAFE_Z,
        on_progress=on_progress,
    )
    assert seen == [
        (1, 4, 0.0, 0.0),
        (2, 4, 10.0, 0.0),
        (3, 4, 10.0, 10.0),
        (4, 4, 0.0, 10.0),
    ]


@pytest.mark.asyncio
async def test_failed_probe_raises_and_retracts():
    driver = FakeDriver(fail_at=(10.0, 0.0))
    with pytest.raises(BedProbeError, match="did not trigger"):
        await probe_bed_mesh(
            cast(Driver, driver),
            x0=0.0,
            y0=0.0,
            dx=10.0,
            dy=10.0,
            nx=2,
            ny=2,
            feed_rate_mm_min=100,
            max_travel_mm=10.0,
            safe_z_mm=SAFE_Z,
        )
    # The final retract happened despite the failure.
    assert driver.moves[-1] == (10.0, 0.0, SAFE_Z)


@pytest.mark.asyncio
async def test_abort_between_points():
    driver = FakeDriver()
    calls = {"n": 0}

    def should_abort():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(BedProbeAborted):
        await probe_bed_mesh(
            cast(Driver, driver),
            x0=0.0,
            y0=0.0,
            dx=10.0,
            dy=10.0,
            nx=3,
            ny=3,
            feed_rate_mm_min=100,
            max_travel_mm=10.0,
            safe_z_mm=SAFE_Z,
            should_abort=should_abort,
        )
    # Aborted mid-run, well before all 9 points.
    assert len(driver.moves) < 9


@pytest.mark.asyncio
async def test_rejects_degenerate_grid():
    driver = FakeDriver()
    with pytest.raises(BedProbeError):
        await probe_bed_mesh(
            cast(Driver, driver),
            x0=0.0,
            y0=0.0,
            dx=10.0,
            dy=10.0,
            nx=1,
            ny=1,
            feed_rate_mm_min=100,
            max_travel_mm=10.0,
            safe_z_mm=SAFE_Z,
        )
    assert driver.moves == []


@pytest.mark.asyncio
async def test_driver_error_wrapped():
    class ExplodingDriver(FakeDriver):
        async def run_probe_cycle(self, axis, max_travel, feed_rate):
            raise RuntimeError("boom")

    with pytest.raises(BedProbeError, match="boom"):
        await probe_bed_mesh(
            cast(Driver, ExplodingDriver()),
            x0=0.0,
            y0=0.0,
            dx=10.0,
            dy=10.0,
            nx=2,
            ny=2,
            feed_rate_mm_min=100,
            max_travel_mm=10.0,
            safe_z_mm=SAFE_Z,
        )
