# flake8: noqa: E402
"""UI tests for the DocSignalHub."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock

import pytest
from blinker import Signal

from rayforge.ui_gtk.sim3d.doc_signals import DocSignalHub


def _make_hub(context=None, doc_editor=None):
    viewports = []
    scene_dirty = []
    rendered = []
    refreshed = []

    hub = DocSignalHub(
        context or MagicMock(machine=None),
        doc_editor or MagicMock(),
        set_viewport=lambda vp: viewports.append(vp),
        mark_scene_dirty=lambda: scene_dirty.append(True),
        request_render=lambda: rendered.append(True),
        refresh_scene=lambda: refreshed.append(True),
        get_gl_initialized=lambda: True,
    )
    return hub, viewports, scene_dirty, rendered, refreshed


def _machine():
    machine = MagicMock()
    machine.wcs_updated = Signal()
    machine.changed = Signal()
    machine.work_area = (100.0, 100.0, 100.0, 100.0)
    machine.work_margins = (0.0, 0.0, 0.0, 0.0)
    machine.y_axis_down = False
    machine.x_axis_right = False
    machine.reverse_x_axis = False
    machine.reverse_y_axis = False
    machine.wcs_origin_is_workarea_origin = True
    machine.has_custom_work_area.return_value = False
    machine.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    return machine


def _doc():
    doc = MagicMock()
    doc.active_layer_changed = Signal()
    doc.active_layer = None
    return doc


def _layer():
    layer = MagicMock()
    layer.updated = Signal()
    layer.wcs = None
    return layer


@pytest.mark.ui
def test_connect_subscribes_machine_signals(ui_context_initializer):
    machine = _machine()
    doc_editor = MagicMock()
    doc_editor.doc = _doc()
    hub, _, _, _, _ = _make_hub(
        context=MagicMock(machine=machine), doc_editor=doc_editor
    )
    hub.connect()
    machine.wcs_updated.send(machine)
    assert hub._active_layer_wcs_conn is None


@pytest.mark.ui
def test_connect_active_layer_wcs(ui_context_initializer):
    machine = _machine()
    doc = _doc()
    layer = _layer()
    doc.active_layer = layer
    doc_editor = MagicMock()
    doc_editor.doc = doc

    hub, viewports, _, rendered, _ = _make_hub(
        context=MagicMock(machine=machine), doc_editor=doc_editor
    )
    hub.connect()

    assert hub._active_layer_wcs_conn is not None
    rendered.clear()
    layer.updated.send(layer)
    assert rendered == [True]
    assert len(viewports) >= 1


@pytest.mark.ui
def test_rotary_enabled_false_without_layer(ui_context_initializer):
    doc = _doc()
    doc_editor = MagicMock()
    doc_editor.doc = doc
    hub, _, _, _, _ = _make_hub(doc_editor=doc_editor)
    assert hub.rotary_enabled is False


@pytest.mark.ui
def test_set_machine_refreshes_viewport(ui_context_initializer):
    machine = _machine()
    doc_editor = MagicMock()
    doc_editor.doc = _doc()
    hub, viewports, _, _, refreshed = _make_hub(
        context=MagicMock(machine=machine), doc_editor=doc_editor
    )
    hub.connect()
    hub.set_machine()
    assert viewports
    assert refreshed == [True]


@pytest.mark.ui
def test_disconnect_unsubscribes_machine(ui_context_initializer):
    machine = _machine()
    doc_editor = MagicMock()
    doc_editor.doc = _doc()
    hub, _, scene_dirty, _, _ = _make_hub(
        context=MagicMock(machine=machine), doc_editor=doc_editor
    )
    hub.connect()
    scene_dirty.clear()
    hub.disconnect()
    machine.wcs_updated.send(machine)
    assert scene_dirty == []
