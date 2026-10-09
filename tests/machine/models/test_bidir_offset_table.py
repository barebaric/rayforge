"""Tests for the machine's speed-dependent bidirectional scan offset."""

import asyncio
from unittest.mock import MagicMock

import pytest

from rayforge.machine.models.bidir_offset import interpolate_bidir_offset
from rayforge.machine.models.machine import Machine
from rayforge.shared.tasker.manager import TaskManager


async def _settle(task_mgr: TaskManager):
    if not await asyncio.to_thread(task_mgr.wait_until_settled, 2000):
        pytest.fail("Task manager did not become idle in time.")


TABLE = [(1000.0, 0.10), (3000.0, 0.30), (6000.0, 0.45)]


class TestInterpolateBidirOffset:
    def test_empty_table_is_zero(self):
        assert interpolate_bidir_offset([], 2000.0) == 0.0

    def test_single_row_is_constant(self):
        table = [(2000.0, 0.2)]
        assert interpolate_bidir_offset(table, 500.0) == 0.2
        assert interpolate_bidir_offset(table, 9000.0) == 0.2

    def test_exact_rows(self):
        for speed, offset in TABLE:
            assert interpolate_bidir_offset(TABLE, speed) == offset

    def test_linear_between_rows(self):
        assert interpolate_bidir_offset(TABLE, 2000.0) == pytest.approx(0.2)
        assert interpolate_bidir_offset(TABLE, 4500.0) == pytest.approx(0.375)

    def test_clamped_outside_the_table(self):
        assert interpolate_bidir_offset(TABLE, 10.0) == 0.10
        assert interpolate_bidir_offset(TABLE, 99999.0) == 0.45

    def test_negative_offsets(self):
        table = [(1000.0, -0.1), (2000.0, 0.1)]
        assert interpolate_bidir_offset(table, 1500.0) == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("lite_context")
class TestMachineBidirOffsetTable:
    async def test_defaults_to_empty(self, machine: Machine):
        assert machine.bidir_offset_table == []
        assert machine.bidir_offset_for_speed(3000) == 0.0

    async def test_set_sorts_and_signals(self, machine: Machine):
        spy = MagicMock()
        machine.changed.connect(spy)
        machine.set_bidir_offset_table([(3000, 0.3), (1000, 0.1)])
        assert machine.bidir_offset_table == [(1000.0, 0.1), (3000.0, 0.3)]
        spy.assert_called_once_with(machine)

        spy.reset_mock()
        machine.set_bidir_offset_table([(1000.0, 0.1), (3000.0, 0.3)])
        spy.assert_not_called()

    async def test_duplicate_speeds_keep_the_last_row(self, machine: Machine):
        machine.set_bidir_offset_table([(1000, 0.1), (1000, 0.2)])
        assert machine.bidir_offset_table == [(1000.0, 0.2)]

    async def test_lookup_uses_the_table(self, machine: Machine):
        machine.set_bidir_offset_table([(1000, 0.1), (3000, 0.3)])
        assert machine.bidir_offset_for_speed(2000) == pytest.approx(0.2)

    async def test_serialization_roundtrip(
        self, machine: Machine, task_mgr: TaskManager, lite_context
    ):
        await _settle(task_mgr)
        machine.set_bidir_offset_table([(1000, 0.1), (3000, 0.3)])
        data = machine.to_dict()
        assert data["machine"]["bidir_offset_table"] == [
            [1000.0, 0.1],
            [3000.0, 0.3],
        ]
        restored = Machine.from_dict(data, context=lite_context)
        await _settle(task_mgr)
        assert restored.bidir_offset_table == [(1000.0, 0.1), (3000.0, 0.3)]
        await restored.shutdown()

    async def test_old_profiles_load_without_table(
        self, machine: Machine, task_mgr: TaskManager, lite_context
    ):
        await _settle(task_mgr)
        data = machine.to_dict()
        data["machine"].pop("bidir_offset_table", None)
        restored = Machine.from_dict(data, context=lite_context)
        await _settle(task_mgr)
        assert restored.bidir_offset_table == []
        assert "bidir_offset_table" not in restored.extra
        await restored.shutdown()
