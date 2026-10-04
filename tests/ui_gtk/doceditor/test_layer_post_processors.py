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


@pytest.mark.ui
def test_renders_unconditionally_without_mesh(ui_context_initializer):
    """The transformer renders even with nothing probed; the addon
    widget itself banners the missing mesh."""
    _make_machine()
    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)

    expander = _mesh_group(group)
    assert expander.get_title() == "Bed Mesh Correction"
    (widget_group,) = group._group_dicts.keys()
    assert widget_group._no_mesh_banner.get_revealed()
    assert layer.post_processors_dicts == []


@pytest.mark.ui
def test_follows_machine_default_subtitle_and_claim(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 2.0}
    ]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    expander = _mesh_group(group)
    assert expander.get_subtitle() == "Follows the machine default"
    # The widget reflects the machine default's values.
    (widget_group,) = group._group_dicts.keys()
    assert widget_group.transformer.z_offset == 2.0
    assert layer.post_processors_dicts == []

    # First interaction claims an entry seeded from the displayed
    # configuration, then applies the change.
    widget_group.param_changed.send(
        widget_group, key="z_offset", value=3.5, name="test"
    )
    assert layer.post_processors_dicts == [
        {"name": MESH_NAME, "enabled": True, "z_offset": 3.5}
    ]
    _, owned, _source = group._group_dicts[(widget_group)]
    assert owned is not None


@pytest.mark.ui
def test_disable_via_switch_claims_disabled_entry(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 0.0}
    ]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    (widget_group,) = group._group_dicts.keys()
    widget_group.param_changed.send(
        widget_group, key="enabled", value=False, name="test"
    )
    assert layer.post_processors_dicts == [
        {"name": MESH_NAME, "enabled": False, "z_offset": 0.0}
    ]


@pytest.mark.ui
def test_owned_entry_edits_in_place(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    layer = Layer(name="l")
    layer.post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 1.0}
    ]
    group = LayerPostProcessorGroup(layer, editor=None)
    (widget_group,) = group._group_dicts.keys()
    original_dict = layer.post_processors_dicts[0]

    widget_group.param_changed.send(
        widget_group, key="z_offset", value=2.25, name="test"
    )
    assert layer.post_processors_dicts[0] is original_dict
    assert original_dict["z_offset"] == 2.25
    # No duplicate entry was created.
    assert len(layer.post_processors_dicts) == 1


@pytest.mark.ui
def test_reset_to_machine_default_removes_entry(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 2.0}
    ]
    layer = Layer(name="l")
    layer.post_processors_dicts = [
        {"name": MESH_NAME, "enabled": False, "z_offset": 0.0}
    ]
    group = LayerPostProcessorGroup(layer, editor=None)
    (transformer_cls, _owned, _source) = next(
        iter(group._group_dicts.values())
    )
    group._reset_to_default(transformer_cls)
    assert layer.post_processors_dicts == []


@pytest.mark.ui
def test_machine_default_without_mesh_still_renders(
    ui_context_initializer,
):
    """A default can exist without a probed mesh (it will no-op);
    the group still renders and the widget flags the missing mesh."""
    machine = _make_machine()
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 0.0}
    ]
    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    expander = _mesh_group(group)
    assert expander.get_subtitle() == "Follows the machine default"
    (widget_group,) = group._group_dicts.keys()
    assert widget_group._no_mesh_banner.get_revealed()
