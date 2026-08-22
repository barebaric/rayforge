from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import pytest

from rayforge.config import BUILTIN_DEVICES_DIR
from rayforge.machine.device.profile import DeviceProfile
from rayforge.machine.device.profile_diff import (
    DIALECT_SECTION,
    MACHINE_SECTION,
    apply_diffs,
    diff_dialect_with_profile,
    diff_heads_with_profile,
    diff_machine_with_profile,
    find_outdated_profiles,
)
from rayforge.machine.models.dialect import GcodeDialect
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine, Origin
from rayforge.shared import tasker

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


@pytest.fixture
def profile() -> DeviceProfile:
    return DeviceProfile.from_path(BUILTIN_DEVICES_DIR / "sculpfun-icube")


@pytest.fixture
def machine(
    profile: DeviceProfile, context_initializer: "RayforgeContext"
) -> "Machine":
    machine = profile.create_machine(context_initializer)
    tasker.task_mgr.wait_until_settled(5000)
    return machine


def test_content_hash_is_stable_and_sensitive(profile: DeviceProfile):
    assert profile.content_hash() == profile.content_hash()

    changed = dc_replace(
        profile, machine_config=dc_replace(profile.machine_config, has_z=True)
    )
    assert changed.content_hash() != profile.content_hash()

    dialect = dict(profile.dialect_config)
    dialect["laser_on"] = "M500"
    changed = dc_replace(profile, dialect_config=dialect)
    assert changed.content_hash() != profile.content_hash()


def test_create_machine_stamps_provenance(
    profile: DeviceProfile, machine: "Machine"
):
    assert machine.source_profile_id == profile.id
    assert machine.reviewed_profile_hash == profile.content_hash()


def test_provenance_survives_roundtrip(
    profile: DeviceProfile,
    machine: "Machine",
    context_initializer: "RayforgeContext",
):
    data = machine.to_dict(include_frozen_dialect=False)
    restored = Machine.from_dict(data, context=context_initializer)
    assert restored.source_profile_id == machine.source_profile_id
    assert restored.reviewed_profile_hash == machine.reviewed_profile_hash


def test_no_diffs_right_after_creation(
    profile: DeviceProfile, machine: "Machine"
):
    assert diff_machine_with_profile(machine, profile) == []
    assert diff_dialect_with_profile(machine, profile) == []


def test_machine_diff_reports_changed_settings(
    profile: DeviceProfile, machine: "Machine"
):
    changed_cfg = dc_replace(
        profile.machine_config,
        has_z=True,
        max_cut_speed=4242,
        origin=Origin.TOP_LEFT,
    )
    changed = dc_replace(profile, machine_config=changed_cfg)

    diffs = {d.key: d for d in diff_machine_with_profile(machine, changed)}

    assert set(diffs) == {"has_z", "max_cut_speed", "origin"}
    has_z = diffs["has_z"]
    assert has_z.section == MACHINE_SECTION
    assert has_z.current_value is False
    assert has_z.profile_value is True
    assert diffs["max_cut_speed"].current_value == 1000
    assert diffs["max_cut_speed"].profile_value == 4242


def test_unset_profile_fields_are_skipped(
    profile: DeviceProfile, machine: "Machine"
):
    # The machine's acceleration differs from the model default, but
    # the profile does not carry the field (None), so there is nothing
    # to apply.
    machine.acceleration = 9999
    assert diff_machine_with_profile(machine, profile) == []


def test_dialect_diff_reports_changed_fields(
    profile: DeviceProfile, machine: "Machine"
):
    dialect = dict(profile.dialect_config)
    dialect["laser_on"] = "M999"
    dialect["preamble"] = ["G21"]
    changed = dc_replace(profile, dialect_config=dialect)

    diffs = {d.key: d for d in diff_dialect_with_profile(machine, changed)}

    assert diffs["dialect.laser_on"].profile_value == "M999"
    assert diffs["dialect.preamble"].profile_value == ["G21"]
    assert all(d.section == DIALECT_SECTION for d in diffs.values())
    # Labels come from the dialect editor varsets.
    assert "Laser On" in diffs["dialect.laser_on"].path


def test_head_diff_reports_changed_fields(
    profile: DeviceProfile, machine: "Machine"
):
    cfg_heads = profile.machine_config.heads or []
    heads = [dict(h) for h in cfg_heads]
    heads[0]["frame_power_percent"] = 55.0
    changed = dc_replace(
        profile, machine_config=dc_replace(profile.machine_config, heads=heads)
    )

    diffs = {d.key: d for d in diff_heads_with_profile(machine, changed)}

    assert set(diffs) == {"head.0.frame_power_percent"}
    diff = diffs["head.0.frame_power_percent"]
    # The profile stores percent (0-100), the instance keeps 0-1:
    # the iCube's 1% frame power is stored as 1.0 in the yaml.
    assert diff.current_value == pytest.approx(1.0)
    assert diff.profile_value == 55.0
    assert "Laser Head" in diff.path
    assert "Frame Power" in diff.path


def test_head_diff_detects_new_physical_power_fields(
    profile: DeviceProfile, machine: "Machine"
):
    """A profile that gains wavelength_nm/max_power_watts (e.g. after
    an app upgrade populates the device YAML) shows up as reviewable
    diffs against a machine created before the fields existed.

    The iCube profile already carries the fields now, so we strip
    them from the machine head to simulate a pre-upgrade machine and
    verify the diff + apply round-trip.
    """
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.wavelength_nm = 0.0
    head.max_power_watts = 0.0

    diffs = {d.key: d for d in diff_heads_with_profile(machine, profile)}

    assert "head.0.wavelength_nm" in diffs
    assert "head.0.max_power_watts" in diffs
    assert diffs["head.0.wavelength_nm"].current_value == 0.0
    assert diffs["head.0.wavelength_nm"].profile_value == 455
    assert diffs["head.0.max_power_watts"].current_value == 0.0
    assert diffs["head.0.max_power_watts"].profile_value == 3


def test_apply_physical_power_fields(
    profile: DeviceProfile, machine: "Machine"
):
    """Applying the physical-power diffs writes the profile values
    onto the existing head object."""
    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    head.wavelength_nm = 0.0
    head.max_power_watts = 0.0

    diffs = diff_heads_with_profile(machine, profile)
    apply_diffs(machine, profile, diffs)

    head = machine.heads[0]
    assert isinstance(head, LaserHead)
    assert head.wavelength_nm == 455
    assert head.max_power_watts == 3


def test_apply_head_field_keeps_head_object(
    profile: DeviceProfile, machine: "Machine"
):
    cfg_heads = profile.machine_config.heads or []
    heads = [dict(h) for h in cfg_heads]
    heads[0]["frame_power_percent"] = 55.0
    changed = dc_replace(
        profile, machine_config=dc_replace(profile.machine_config, heads=heads)
    )
    diffs = diff_heads_with_profile(machine, changed)
    head_before = machine.heads[0]

    apply_diffs(machine, changed, diffs)

    # The head object itself is updated in place, never replaced.
    from rayforge.machine.models.laser import LaserHead

    assert isinstance(machine.heads[0], LaserHead)
    assert machine.heads[0] is head_before
    assert machine.heads[0].frame_power_percent == pytest.approx(0.55)


def test_apply_machine_settings(profile: DeviceProfile, machine: "Machine"):
    changed_cfg = dc_replace(
        profile.machine_config,
        has_z=False,
        max_cut_speed=4242,
        axis_extents=(100.0, 100.0),
    )
    changed = dc_replace(profile, machine_config=changed_cfg)
    diffs = diff_machine_with_profile(machine, changed)

    apply_diffs(machine, changed, diffs)

    assert machine.has_z_axis is False
    assert machine.max_cut_speed == 4242
    assert machine.axis_extents == (100.0, 100.0)


def test_apply_dialect_keeps_uid(profile: DeviceProfile, machine: "Machine"):
    dialect = machine.dialect
    assert dialect is not None and dialect.is_custom
    uid = dialect.uid

    changed_dict = dict(profile.dialect_config)
    changed_dict["laser_on"] = "M999"
    changed = dc_replace(profile, dialect_config=changed_dict)
    diffs = diff_dialect_with_profile(machine, changed)

    apply_diffs(machine, changed, diffs)

    updated = machine.dialect
    assert updated is not None
    assert updated.uid == uid
    assert updated.laser_on == "M999"


def test_find_outdated_profiles(profile: DeviceProfile, machine: "Machine"):
    profiles_by_id = {profile.id: profile}

    assert find_outdated_profiles([machine], profiles_by_id) == []

    changed = dc_replace(
        profile,
        machine_config=dc_replace(profile.machine_config, max_cut_speed=1),
    )
    profiles_by_id = {changed.id: changed}
    assert find_outdated_profiles([machine], profiles_by_id) == [
        (machine, changed)
    ]


def test_find_outdated_ignores_unlinked_machines(
    context_initializer: "RayforgeContext", profile: DeviceProfile
):
    from rayforge.machine.models.machine import Machine

    bare = Machine(context_initializer)
    assert find_outdated_profiles([bare], {profile.id: profile}) == []


def test_find_outdated_includes_never_reviewed(
    profile: DeviceProfile, machine: "Machine"
):
    """A machine with a source_profile_id but no reviewed_profile_hash
    (created before the profile-review system existed) is included so
    its first review can pick up new profile settings."""
    machine.reviewed_profile_hash = None
    profiles_by_id = {profile.id: profile}
    assert find_outdated_profiles([machine], profiles_by_id) == [
        (machine, profile)
    ]


def test_builtin_profiles_have_stable_ids():
    for d in sorted(BUILTIN_DEVICES_DIR.iterdir()):
        if not d.is_dir():
            continue
        profile = DeviceProfile.from_path(d)
        assert profile.id == d.name


def test_gcode_dialect_roundtrip_template(profile: DeviceProfile):
    """
    The template roundtrip covers every field a profile can carry, so
    the dialect diff never misses a profile-defined setting.
    """
    dialect = GcodeDialect.from_template_dict(
        profile.dialect_config, label="x", description="", is_custom=True
    )
    assert set(profile.dialect_config) <= set(dialect.to_template_dict())
