"""UI tests for the layer post processor settings group."""

import pytest
from gi.repository import Adw

from rayforge.core.layer import Layer
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.doceditor.layer_post_processors import (
    LayerPostProcessorGroup,
)

MESH_DICT = {"name": "MeshCorrectionTransformer", "enabled": True}


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


@pytest.mark.ui
def test_group_shows_machine_default_row(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [dict(MESH_DICT)]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    assert group._group_dicts == {}

    # The machine-defaults info row is present with its actions.
    from gi.repository import Adw

    action_rows = [
        child for child in group._children if isinstance(child, Adw.ActionRow)
    ]
    assert action_rows


@pytest.mark.ui
def test_group_disable_default_writes_layer_dict(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [dict(MESH_DICT)]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    group._disable_default("MeshCorrectionTransformer")

    assert layer.post_processors_dicts == [
        {"name": "MeshCorrectionTransformer", "enabled": False}
    ]
    # Rebuilt: the layer now owns a disabled entry, so the info row
    # for unoverridden defaults is gone.
    assert group._group_dicts


@pytest.mark.ui
def test_group_customize_copies_machine_default(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": "MeshCorrectionTransformer", "enabled": True, "z_offset": 2.0}
    ]

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    group._override_default("MeshCorrectionTransformer")

    assert layer.post_processors_dicts == [
        {"name": "MeshCorrectionTransformer", "enabled": True, "z_offset": 2.0}
    ]


@pytest.mark.ui
def test_group_renders_layer_owned_entry(ui_context_initializer):
    """With the addon registries populated, the layer's dict renders
    as a settings group and param changes persist to the layer."""
    machine = _make_machine()
    _make_mesh(machine)

    layer = Layer(name="l")
    layer.post_processors_dicts = [dict(MESH_DICT)]
    group = LayerPostProcessorGroup(layer, editor=None)
    assert group._group_dicts, (
        "expected a widget group for the mesh correction entry"
    )

    ((widget_group, _t_dict),) = group._group_dicts.items()
    widget_group.param_changed.send(
        widget_group, key="z_offset", value=1.25, name="test"
    )
    assert layer.post_processors_dicts[0]["z_offset"] == 1.25
    assert layer.post_processors_dicts[0]["name"] == (
        "MeshCorrectionTransformer"
    )


def _action_row_titles(group) -> list[str]:
    from gi.repository import Adw

    return [
        child.get_title()
        for child in group._children
        if isinstance(child, Adw.ActionRow)
    ]


@pytest.mark.ui
def test_empty_group_offers_add_with_probed_mesh(ui_context_initializer):
    """No defaults, no layer entries, mesh probed: an Add row shows."""
    machine = _make_machine()
    _make_mesh(machine)

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    assert "Bed Mesh Correction" in _action_row_titles(group)

    (add_row,) = [
        child
        for child in group._children
        if isinstance(child, Adw.ActionRow)
        and child.get_title() == "Bed Mesh Correction"
    ] or [None]
    assert add_row is not None
    group._add_mesh_correction(None)
    assert layer.post_processors_dicts == [
        {"name": "MeshCorrectionTransformer", "enabled": True, "z_offset": 0.0}
    ]
    # Rebuilt: the layer now owns an entry rendered as a widget group.
    assert group._group_dicts


@pytest.mark.ui
def test_empty_group_shows_hint_without_mesh(ui_context_initializer):
    """No defaults, no layer entries, no mesh: a hint row explains."""
    _make_machine()

    layer = Layer(name="l")
    group = LayerPostProcessorGroup(layer, editor=None)
    assert "No Post Processing Available" in _action_row_titles(group)


@pytest.mark.ui
def test_add_is_idempotent_when_entry_exists(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)

    layer = Layer(name="l")
    layer.post_processors_dicts = [dict(MESH_DICT)]
    group = LayerPostProcessorGroup(layer, editor=None)
    before = len(group._children)
    group._add_mesh_correction(None)
    assert len(layer.post_processors_dicts) == 1
    assert len(group._children) == before
