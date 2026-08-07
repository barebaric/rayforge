"""UI tests for the shared ThemeColorService."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock

import numpy as np
import pytest
from blinker import Signal

from rayforge.core.color import ColorSet
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.shared.gtk_color import GtkColorResolver
from rayforge.ui_gtk.shared.theme_service import ThemeColorService


def _theme_color_set() -> ColorSet:
    return ColorSet(
        {
            "cut": (1.0, 0.0, 0.0, 1.0),
            "engrave": np.full((256, 4), 0.5, dtype=np.float32),
            "travel": (0.0, 1.0, 0.0, 1.0),
            "zero_power": (0.0, 0.0, 1.0, 1.0),
        }
    )


def _bind_service(service, widget):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        return service.color_set


@pytest.mark.ui
def test_unbound_service_returns_no_colors(ui_context_initializer):
    service, _ = ThemeColorService(), MagicMock()
    assert service.color_set is None
    assert service.laser_color_sets == {}
    assert service.layer_color_sets == {}


@pytest.mark.ui
def test_bind_marks_dirty_and_refreshes(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    color_set = _bind_service(service, widget)
    assert color_set is not None
    assert service.dirty is False


@pytest.mark.ui
def test_on_style_changed_marks_dirty(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    _bind_service(service, widget)
    service._on_style_changed(widget, None)
    assert service.dirty is True


@pytest.mark.ui
def test_set_machine_resolves_laser_sets(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    laser = LaserHead()
    laser.set_cut_color("#ff0000")
    laser.set_raster_color("#00ff00")
    machine = MagicMock(heads=[laser, MagicMock(uid="spindle")])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        service.set_machine(machine)
        laser_sets = service.laser_color_sets

    assert "spindle" not in laser_sets
    assert laser.uid in laser_sets


@pytest.mark.ui
def test_color_lut_provider_uses_service_sets(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        provider = service.color_lut_provider()

    assert provider is not None
    assert provider.cut_lut().shape == (256, 4)


@pytest.mark.ui
def test_color_lut_provider_cached_until_dirty(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        first = service.color_lut_provider()
        second = service.color_lut_provider()
        assert second is first
        service.mark_dirty()
        rebuilt = service.color_lut_provider()
        assert rebuilt is not first


@pytest.mark.ui
def test_set_machine_same_instance_no_extra_dirty(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    machine = MagicMock(heads=[])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        service.set_machine(machine)
        service.color_set  # trigger refresh, clears dirty
        service.set_machine(machine)

    assert service.dirty is False


@pytest.mark.ui
def test_machine_change_marks_dirty(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    machine = MagicMock(heads=[])
    machine.changed = Signal()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        service.set_machine(machine)
        service.color_set  # trigger refresh, clears dirty
        assert service.dirty is False
        machine.changed.send(machine)

    assert service.dirty is True


@pytest.mark.ui
def test_doc_update_marks_dirty(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    doc = MagicMock()
    doc.descendant_updated = Signal()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        service.set_doc(doc)
        service.color_set  # trigger refresh, clears dirty
        assert service.dirty is False
        doc.descendant_updated.send(doc)

    assert service.dirty is True


@pytest.mark.ui
def test_switching_machine_disconnects_old_signal(ui_context_initializer):
    service, widget = ThemeColorService(), MagicMock()
    old_machine = MagicMock(heads=[])
    old_machine.changed = Signal()
    new_machine = MagicMock(heads=[])
    new_machine.changed = Signal()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            GtkColorResolver, "resolve", lambda self, spec: _theme_color_set()
        )
        service.bind(widget)
        service.set_machine(old_machine)
        service.set_machine(new_machine)
        service.color_set  # trigger refresh, clears dirty
        old_machine.changed.send(old_machine)

    assert service.dirty is False
    new_machine.changed.send(new_machine)
    assert service.dirty is True
