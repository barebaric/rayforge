"""
Tests for the WCS dropdown model synchronization in BottomPanel.

Switching drivers or updating offsets used to re-emit items_changed on
an unchanged list model, which violates the list model contract and
crashes GTK with an assertion while it is setting up factory widgets
(issue #415).
"""

from typing import Any

import pytest
from gi.repository import Gtk

from rayforge.machine.models.coordinate_system import CoordinateSystem


@pytest.fixture
def wcs_panel(sync_machine):
    """A BottomPanel instance with Gtk initialization bypassed, wired to
    a real machine and a populated WCS model."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    panel: Any = BottomPanel.__new__(BottomPanel)
    panel.machine = sync_machine
    panel.wcs_list = list(sync_machine.supported_wcs)
    panel._wcs_model = Gtk.StringList.new(panel.wcs_list)
    panel._wcs_model_state = panel._get_wcs_model_state()
    return panel


def _model_strings(panel) -> list[str]:
    model = panel._wcs_model
    return [model.get_string(i) for i in range(model.get_n_items())]


def _record_items_changed(panel) -> list[tuple[int, int, int]]:
    emissions: list[tuple[int, int, int]] = []
    panel._wcs_model.connect(
        "items-changed",
        lambda _m, position, removed, added: emissions.append(
            (position, removed, added)
        ),
    )
    return emissions


@pytest.mark.ui
def test_sync_wcs_model_noop_on_unchanged_state(wcs_panel):
    """Repeated updates must not touch the model when nothing changed."""
    emissions = _record_items_changed(wcs_panel)

    wcs_panel._sync_wcs_model()
    wcs_panel._sync_wcs_model()

    assert emissions == []
    assert _model_strings(wcs_panel) == wcs_panel.machine.supported_wcs


@pytest.mark.ui
def test_sync_wcs_model_splices_on_offset_change(wcs_panel):
    """Changed offsets must arrive as one real splice of the model."""
    wcs = wcs_panel.machine.supported_wcs[0]
    wcs_panel.machine.update_wcs_offset(wcs, (1.0, 2.0, 3.0))
    emissions = _record_items_changed(wcs_panel)

    wcs_panel._sync_wcs_model()

    assert len(emissions) == 1
    position, removed, added = emissions[0]
    assert (position, removed, added) == (
        0,
        len(wcs_panel.machine.supported_wcs),
        len(wcs_panel.machine.supported_wcs),
    )
    assert _model_strings(wcs_panel) == wcs_panel.machine.supported_wcs
    assert wcs_panel.wcs_list == wcs_panel.machine.supported_wcs


@pytest.mark.ui
def test_sync_wcs_model_splices_on_systems_change(wcs_panel):
    """A driver reporting a different WCS set must be reflected."""
    machine = wcs_panel.machine
    machine.coordinate_systems = {
        name: CoordinateSystem(name=name) for name in ("G54", "G55")
    }
    emissions = _record_items_changed(wcs_panel)

    wcs_panel._sync_wcs_model()

    assert len(emissions) == 1
    assert _model_strings(wcs_panel) == ["G54", "G55"]
    assert wcs_panel.wcs_list == ["G54", "G55"]
