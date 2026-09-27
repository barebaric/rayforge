"""UI tests for the profile review dialog."""

from dataclasses import replace as dc_replace

import pytest
from gi.repository import Adw

from rayforge.config import BUILTIN_DEVICES_DIR
from rayforge.machine.device.profile import DeviceProfile
from rayforge.machine.device.profile_diff import (
    DEVICE_NOTES_SECTION,
    HEADS_SECTION,
    diff_device_notes_with_profile,
    diff_dialect_with_profile,
    diff_heads_with_profile,
    diff_machine_with_profile,
)
from rayforge.machine.models.machine import Machine
from rayforge.shared import tasker
from rayforge.ui_gtk.machine.profile_review_dialog import ProfileReviewDialog


@pytest.fixture
def machine(ui_context_initializer) -> Machine:
    profile = DeviceProfile.from_path(
        BUILTIN_DEVICES_DIR / "sculpfun-icube-3w"
    )
    machine = profile.create_machine(ui_context_initializer)
    tasker.task_mgr.wait_until_settled(5000)
    return machine


def _changed_profile(profile: DeviceProfile) -> DeviceProfile:
    dialect = dict(profile.dialect_config)
    dialect["laser_on"] = "M999"
    heads = [dict(h) for h in profile.machine_config.heads or []]
    if heads:
        heads[0]["frame_power_percent"] = 55.0
    return dc_replace(
        profile,
        machine_config=dc_replace(
            profile.machine_config, max_cut_speed=4242, heads=heads
        ),
        dialect_config=dialect,
    )


@pytest.mark.ui
def test_dialog_builds_one_switch_row_per_diff(machine):
    profile = _changed_profile(
        DeviceProfile.from_path(BUILTIN_DEVICES_DIR / "sculpfun-icube-3w")
    )
    expected = (
        len(diff_machine_with_profile(machine, profile))
        + len(diff_heads_with_profile(machine, profile))
        + len(diff_dialect_with_profile(machine, profile))
    )

    dialog = ProfileReviewDialog(machine, profile)

    assert len(dialog._rows) == expected
    for diff, row in dialog._rows:
        assert isinstance(row, Adw.SwitchRow)
        assert row.get_active() is True
        # The row title carries the setting path and the subtitle both
        # values.
        assert diff.path in row.get_title()
        subtitle = row.get_subtitle() or ""
        assert "→" in subtitle
    # Head changes are offered as rows too.
    assert any(diff.section == HEADS_SECTION for diff, _ in dialog._rows)


@pytest.mark.ui
def test_dialog_apply_applies_selected_and_marks_reviewed(machine):
    profile = _changed_profile(
        DeviceProfile.from_path(BUILTIN_DEVICES_DIR / "sculpfun-icube-3w")
    )
    closed_callback_called = []

    dialog = ProfileReviewDialog(
        machine, profile, on_closed=lambda: closed_callback_called.append(1)
    )
    # Deselect all but one row.
    dialog._rows[1][1].set_active(False)

    dialog._on_apply_clicked(dialog._rows[0][1])

    applied_keys = {diff.key for diff, _ in [dialog._rows[0]]}
    if "max_cut_speed" in applied_keys:
        assert machine.max_cut_speed == 4242
    else:
        assert machine.dialect.laser_on == "M999"
    assert machine.reviewed_profile_hash == profile.content_hash()
    assert closed_callback_called


@pytest.mark.ui
def test_dialog_ignore_marks_reviewed_without_applying(machine):
    profile = _changed_profile(
        DeviceProfile.from_path(BUILTIN_DEVICES_DIR / "sculpfun-icube-3w")
    )
    old_speed = machine.max_cut_speed

    dialog = ProfileReviewDialog(machine, profile)
    dialog._on_ignore_clicked(None)

    assert machine.max_cut_speed == old_speed
    assert machine.reviewed_profile_hash == profile.content_hash()


@pytest.mark.ui
def test_dialog_offers_profile_notes_as_separately_selectable_diff(machine):
    base = DeviceProfile.from_path(BUILTIN_DEVICES_DIR / "sculpfun-icube-3w")
    profile = dc_replace(
        base, meta=dc_replace(base.meta, notes="New camera guidance.")
    )
    machine.user_notes = "Private alignment notes."

    dialog = ProfileReviewDialog(machine, profile)

    notes_rows = [
        (diff, row)
        for diff, row in dialog._rows
        if diff.section == DEVICE_NOTES_SECTION
    ]
    assert len(notes_rows) == len(
        diff_device_notes_with_profile(machine, profile)
    )
    diff, row = notes_rows[0]
    assert diff.profile_value == "New camera guidance."
    assert row.get_active()

    row.set_active(False)
    dialog._apply(dialog._selected_diffs())
    assert machine.device_notes is None
    assert machine.user_notes == "Private alignment notes."
