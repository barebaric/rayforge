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


def _mesh(group):
    (widget_group,) = group._group_dicts.keys()
    return widget_group, group._group_dicts[widget_group]


@pytest.mark.ui
def test_renders_registered_transformers_unconditionally(
    ui_context_initializer,
):
    """The group shows the mesh transformer with no entry and no
    mesh probed; the widget reports the missing mesh itself."""
    machine = _make_machine()
    group = MachinePostProcessorsGroup(machine)

    widget_group, (cls, entry, source) = _mesh(group)
    assert cls.__name__ == MESH_NAME
    assert entry is None
    assert source["name"] == MESH_NAME
    assert "No bed mesh has been probed" in (
        widget_group.get_description() or ""
    )


@pytest.mark.ui
def test_claim_writes_machine_default(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    group = MachinePostProcessorsGroup(machine)

    widget_group, (cls, _entry, _source) = _mesh(group)
    assert cls.__name__ == MESH_NAME
    widget_group.param_changed.send(
        widget_group, key="enabled", value=False, name="test"
    )

    assert machine.default_post_processors_dicts == [
        {"name": MESH_NAME, "enabled": False}
    ]
    _, entry, _source = group._group_dicts[widget_group]
    assert entry is not None
    assert entry["enabled"] is False


@pytest.mark.ui
def test_edit_existing_entry_persists(ui_context_initializer):
    machine = _make_machine()
    _make_mesh(machine)
    machine.default_post_processors_dicts = [
        {"name": MESH_NAME, "enabled": True}
    ]
    group = MachinePostProcessorsGroup(machine)

    widget_group, (_cls, _entry, _source) = _mesh(group)
    widget_group.param_changed.send(
        widget_group, key="enabled", value=False, name="test"
    )
    assert machine.default_post_processors_dicts
    assert machine.default_post_processors_dicts[0]["enabled"] is False
    _, entry, _source = group._group_dicts[widget_group]
    assert entry is not None
    assert entry["enabled"] is False
