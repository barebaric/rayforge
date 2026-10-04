"""UI tests for the layer post processor settings group."""

import pytest
from gi.repository import Adw

from rayforge.core.layer import Layer
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.doceditor.layer_post_processors import (
    LayerPostProcessorGroup,
)

MESH_NAME = "MeshCorrectionTransformer"


def _make_machine() -> Machine:
    from rayforge.context import get_context

    machine = Machine(get_context())
    get_context().machine_mgr.add_machine(machine)
    get_context().config.machine = machine
    return machine


def _make_mesh(machine: Machine):
    from rayforge.machine.models.bed_mesh import BedMesh

    mesh = BedMesh()
    mesh.set_params(
        x0=0.0,
        y0=0.0,
        dx=10.0,
        dy=10.0,
        nx=2,
        ny=2,
        heights=[0.0, 0.0, 0.0, 0.0],
    )
    machine.set_bed_mesh(mesh)


def _mesh_group(group) -> Adw.ExpanderRow:
    """The single rendered expander for the mesh transformer."""
    (expander,) = [
        child
        for child in group._children
        if isinstance(child, Adw.ExpanderRow)
    ]
    return expander


def _mesh_widget(group):
    (widget_group,) = group._group_dicts.keys()
    return widget_group


@pytest.mark.ui
def test_renders_unconditionally_without_mesh(ui_context_initializer):
    """The transformer renders even with nothing probed; the addon
    widget reports the missing mesh in its description."""
    _make_machine()
    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)

    expander = _mesh_group(group)
    assert expander.get_title() == "Bed Mesh Correction"
    assert "No bed mesh has been probed" in (
        _mesh_widget(group).get_description() or ""
    )
    assert layer.post_processors_dicts == []


@pytest.mark.ui
def test_follows_machine_default_and_claims_on_edit(
    ui_context_initializer,
):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True}
    ]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    assert _mesh_group(group).get_subtitle() == "Follows the machine default"
    assert layer.post_processors_dicts == []

    _mesh_widget(group).param_changed.send(
        _mesh_widget(group), key="enabled", value=False, name="test"
    )
    assert layer.post_processors_dicts == [
        {"name": MESH_NAME, "enabled": False}
    ]
    _cls, owned, _source = next(iter(group._group_dicts.values()))
    assert owned is not None


@pytest.mark.ui
def test_owned_entry_edits_in_place(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    layer = Layer(name="l")
    layer.post_processors_dicts = [{"name": MESH_NAME, "enabled": True}]
    group = LayerPostProcessorGroup(layer, editor=None)
    original_dict = layer.post_processors_dicts[0]

    _mesh_widget(group).param_changed.send(
        _mesh_widget(group), key="enabled", value=False, name="test"
    )
    assert layer.post_processors_dicts[0] is original_dict
    assert original_dict["enabled"] is False
    assert len(layer.post_processors_dicts) == 1


@pytest.mark.ui
def test_reset_to_machine_default_removes_entry(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True}
    ]
    layer = Layer(name="l")
    layer.post_processors_dicts = [{"name": MESH_NAME, "enabled": False}]
    group = LayerPostProcessorGroup(layer, editor=None)
    transformer_cls = next(iter(group._group_dicts.values()))[0]
    group._reset_to_default(transformer_cls)
    assert layer.post_processors_dicts == []
