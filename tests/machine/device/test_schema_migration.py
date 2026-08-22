"""Tests for the schema-version migration system.

Verifies that machines saved by older app versions (schema_version 0)
get the new physical-power settings surfaced as reviewable diffs, and
that applying the diffs writes the suggested values onto the head.
"""

import pytest

from rayforge.machine.device.schema_migration import (
    CURRENT_SCHEMA_VERSION,
    apply_schema_migrations,
    find_schema_migrations,
)
from rayforge.machine.models.laser import LaserHead, LaserType
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead


@pytest.fixture
def old_machine(lite_context) -> Machine:
    """A machine with schema_version 0 (pre-physical-power) and a
    diode laser head whose wavelength/power are still at the 0
    sentinel."""
    machine = Machine(lite_context)
    machine.schema_version = 0
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.laser_type = LaserType.DIODE
    head.wavelength_nm = 0.0
    head.max_power_watts = 0.0
    return machine


def test_current_schema_version_is_positive():
    assert CURRENT_SCHEMA_VERSION >= 1


def test_find_migrations_returns_wavelength_and_power(old_machine):
    diffs = {d.key: d for d in find_schema_migrations(old_machine)}
    assert "head.0.wavelength_nm" in diffs
    assert "head.0.max_power_watts" in diffs


def test_migration_suggests_laser_type_default_wavelength(old_machine):
    diffs = {d.key: d for d in find_schema_migrations(old_machine)}
    # DIODE default wavelength is 455 nm
    assert diffs["head.0.wavelength_nm"].current_value == 0.0
    assert diffs["head.0.wavelength_nm"].profile_value == 455.0


def test_migration_suggests_default_optical_power(old_machine):
    diffs = {d.key: d for d in find_schema_migrations(old_machine)}
    # Default is 40 W
    assert diffs["head.0.max_power_watts"].current_value == 0.0
    assert diffs["head.0.max_power_watts"].profile_value == 40.0


def test_migration_suggests_co2_wavelength(lite_context):
    machine = Machine(lite_context)
    machine.schema_version = 0
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.laser_type = LaserType.CO2
    head.wavelength_nm = 0.0
    head.max_power_watts = 0.0

    diffs = {d.key: d for d in find_schema_migrations(machine)}
    assert diffs["head.0.wavelength_nm"].profile_value == 10600.0


def test_no_migrations_for_current_schema(old_machine):
    old_machine.schema_version = CURRENT_SCHEMA_VERSION
    assert find_schema_migrations(old_machine) == []


def test_no_migrations_when_values_already_set(old_machine):
    old_machine.heads[0].wavelength_nm = 455.0
    old_machine.heads[0].max_power_watts = 20.0
    diffs = find_schema_migrations(old_machine)
    assert diffs == []


def test_apply_migrations_writes_values_and_stamps_version(old_machine):
    diffs = find_schema_migrations(old_machine)
    apply_schema_migrations(old_machine, diffs)

    head = old_machine.heads[0]
    assert head.wavelength_nm == 455.0
    assert head.max_power_watts == 40.0
    assert old_machine.schema_version == CURRENT_SCHEMA_VERSION


def test_apply_partial_migrations_stamps_version(old_machine):
    # Apply only the wavelength diff
    diffs = find_schema_migrations(old_machine)
    wavelength_diff = [d for d in diffs if "wavelength_nm" in d.key]
    apply_schema_migrations(old_machine, wavelength_diff)

    head = old_machine.heads[0]
    assert head.wavelength_nm == 455.0
    assert head.max_power_watts == 0.0  # not applied
    # Version is stamped regardless of how many were selected
    assert old_machine.schema_version == CURRENT_SCHEMA_VERSION


def test_schema_version_roundtrips_through_serialization(
    lite_context,
):
    machine = Machine(lite_context)
    machine.schema_version = 3
    data = machine.to_dict()
    assert data["machine"]["schema_version"] == 3

    restored = Machine.from_dict(data, context=lite_context)
    assert restored.schema_version == 3


def test_schema_version_defaults_to_zero_for_old_data(lite_context):
    data = {"machine": {"name": "Old", "driver": "GrblSerialDriver"}}
    machine = Machine.from_dict(data, context=lite_context)
    assert machine.schema_version == 0


def test_migration_ignores_non_laser_heads(lite_context):
    machine = Machine(lite_context)
    machine.schema_version = 0
    machine.heads = [SpindleHead()]
    assert find_schema_migrations(machine) == []


def test_no_double_prompt_when_profile_filled_some_values(
    lite_context,
):
    """When a profile review fills some (but not all) new settings,
    the schema migration only prompts for the ones still at the
    sentinel — not the ones the profile already set.

    Scenario: schema v1 adds wavelength + optical power. The profile
    provides wavelength (455) but not optical power. After the profile
    review, the schema migration should prompt only for optical power.
    """
    machine = Machine(lite_context)
    machine.schema_version = 0
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    # Profile review set wavelength but not optical power:
    head.wavelength_nm = 455.0
    head.max_power_watts = 0.0

    diffs = {d.key: d for d in find_schema_migrations(machine)}
    assert "head.0.wavelength_nm" not in diffs  # filled by profile
    assert "head.0.max_power_watts" in diffs  # still at sentinel
    assert diffs["head.0.max_power_watts"].current_value == 0.0
    assert diffs["head.0.max_power_watts"].profile_value == 40.0


def test_no_prompt_when_profile_filled_all_values(lite_context):
    """When a profile review fills all new settings, the schema
    migration has nothing left to prompt — no diffs at all, even
    though schema_version is still 0."""
    machine = Machine(lite_context)
    machine.schema_version = 0
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.wavelength_nm = 455.0
    head.max_power_watts = 20.0

    assert find_schema_migrations(machine) == []


def test_no_prompt_when_value_differs_from_default(lite_context):
    """A setting set to a non-default, non-sentinel value (e.g.
    wavelength 450 instead of the 455 default) is not re-prompted.
    The migration only targets the 0 sentinel, not 'differs from
    suggested'."""
    machine = Machine(lite_context)
    machine.schema_version = 0
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.wavelength_nm = 450.0  # not the 455 default, but not 0
    head.max_power_watts = 0.0

    diffs = {d.key: d for d in find_schema_migrations(machine)}
    assert "head.0.wavelength_nm" not in diffs
    assert "head.0.max_power_watts" in diffs


def test_profile_machine_not_skipped(lite_context):
    """A machine with a source_profile_id is NOT skipped by the
    schema migration — the profile may not cover all new settings.
    The schema migration still runs, but only surfaces settings
    still at the sentinel (which the profile review would have
    filled if the profile provided them)."""
    machine = Machine(lite_context)
    machine.schema_version = 0
    machine.source_profile_id = "sculpfun-icube"
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.wavelength_nm = 0.0
    head.max_power_watts = 0.0

    diffs = find_schema_migrations(machine)
    # The schema migration still fires for profile-linked machines:
    assert len(diffs) == 2
