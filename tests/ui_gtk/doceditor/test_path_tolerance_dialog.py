# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.ui_gtk.doceditor.path_tolerance_dialog import (
    DEFAULT_TOLERANCE_MM,
    PathToleranceDialog,
)


@pytest.fixture(autouse=True)
def _reset_remembered_values(ui_context_initializer):
    PathToleranceDialog._last_values.clear()
    yield
    PathToleranceDialog._last_values.clear()


def _dialog(calls: list[float], key="close-paths") -> PathToleranceDialog:
    return PathToleranceDialog(
        heading="Close Paths",
        body="Body",
        apply_label="Close",
        key=key,
        on_apply=calls.append,
    )


@pytest.mark.ui
def test_apply_runs_callback_with_tolerance():
    calls: list[float] = []
    dialog = _dialog(calls)
    assert dialog.get_tolerance_mm() == pytest.approx(DEFAULT_TOLERANCE_MM)

    dialog.tolerance_row.set_value_in_base_units(0.25)
    dialog.emit("response", "apply")

    assert calls == [pytest.approx(0.25)]


@pytest.mark.ui
def test_cancel_does_not_run_callback():
    calls: list[float] = []
    dialog = _dialog(calls)
    dialog.emit("response", "cancel")
    assert calls == []


@pytest.mark.ui
def test_last_value_is_remembered_per_operation():
    first = _dialog([])
    first.tolerance_row.set_value_in_base_units(0.5)
    first.emit("response", "apply")

    assert _dialog([]).get_tolerance_mm() == pytest.approx(0.5)
    other = _dialog([], key="join-paths")
    assert other.get_tolerance_mm() == pytest.approx(DEFAULT_TOLERANCE_MM)
