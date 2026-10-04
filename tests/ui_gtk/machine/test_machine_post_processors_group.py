"""UI tests for the machine-level default post processors group."""

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.machine_post_processors_group import (
    MachinePostProcessorsGroup,
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


def _mesh_group(group) -> tuple:
    (key,) = group._group_dicts.keys()
    return key, group._group_dicts[key]


@pytest.mark.ui
def test_renders_registered_transformers_unconditionally(
    ui_context_initializer,
):
    """The group shows the mesh transformer with no entry and no
    mesh probed; the widget flags the missing mesh itself."""
    machine = _make_machine()
    group = MachinePostProcessorsGroup(machine)

    widget_group, (cls, entry, source) = _mesh_group(group)
    assert cls.__name__ == MESH_NAME
    assert entry is None
    assert source["name"] == MESH_NAME
    assert widget_group._no_mesh_banner.get_revealed()


@pytest.mark.ui
def test_claim_writes_machine_default(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    group = MachinePostProcessorsGroup(machine)

    widget_group, (cls, _entry, _source) = _mesh_group(group)
    assert cls.__name__ == MESH_NAME
    widget_group.param_changed.send(
        widget_group, key="z_offset", value=1.5, name="test"
    )

    assert machine.default_post_processors_dicts == [
        {"name": MESH_NAME, "enabled": True, "z_offset": 1.5}
    ]
    _, entry, _source = group._group_dicts[widget_group]
    assert entry is not None
    assert entry["z_offset"] == 1.5


@pytest.mark.ui
def test_edit_existing_entry_persists(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True, "z_offset": 0.5}
    ]
    group = MachinePostProcessorsGroup(machine)

    widget_group, (_cls, entry, _source) = _mesh_group(group)
    widget_group.param_changed.send(
        widget_group, key="z_offset", value=2.5, name="test"
    )
    assert machine.default_post_processors_dicts
    assert machine.default_post_processors_dicts[0]["z_offset"] == 2.5
    # The group tracks the fresh dict object written by the machine.
    _, entry, _source = group._group_dicts[widget_group]
    assert entry is not None
    assert entry["z_offset"] == 2.5


@pytest.mark.ui
def test_disable_via_switch(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    group = MachinePostProcessorsGroup(machine)

    widget_group, _pair = _mesh_group(group)
    widget_group.param_changed.send(
        widget_group, key="enabled", value=False, name="test"
    )
    assert machine.default_post_processors_dicts == [
        {"name": MESH_NAME, "enabled": False, "z_offset": 0.0}
    ]
