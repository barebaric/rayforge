"""UI tests for the bed-mesh probing page and its settings-dialog gating."""

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.bed_mesh_page import BedMeshPage
from rayforge.ui_gtk.machine.settings_dialog import MachineSettingsDialog


def _make_machine() -> Machine:
    from rayforge.context import get_context

    return Machine(get_context())


@pytest.mark.ui
def test_bed_mesh_page_constructs(ui_context_initializer):
    machine = _make_machine()
    page = BedMeshPage(machine=machine)
    assert page.columns_row.get_value() == 9
    assert page.rows_row.get_value() == 9
    # No mesh probed: the info row says so and no view data is set.
    assert page.mesh_info_row.get_title() == "No mesh probed yet"


@pytest.mark.ui
def test_bed_mesh_page_shows_existing_mesh(ui_context_initializer):
    from rayforge.machine.models.bed_mesh import BedMesh

    machine = _make_machine()
    mesh = BedMesh()
    mesh.set_params(
        x0=0.0,
        y0=0.0,
        dx=10.0,
        dy=10.0,
        nx=2,
        ny=2,
        heights=[0.0, 1.0, 2.0, 3.0],
    )
    machine.set_bed_mesh(mesh)
    page = BedMeshPage(machine=machine)
    assert page.mesh_info_row.get_title() == "2×2 grid, 10.0 mm spacing"
    assert "0.00 … 3.00" in (page.mesh_info_row.get_subtitle() or "")


@pytest.mark.ui
def test_bed_mesh_page_refuses_to_probe_disconnected(
    ui_context_initializer,
):
    machine = _make_machine()
    page = BedMeshPage(machine=machine)
    page._start()
    assert page.status_row.get_subtitle() == "Not connected"


@pytest.mark.ui
def test_settings_dialog_hides_bed_mesh_without_z_axis(
    ui_context_initializer,
):
    machine = _make_machine()
    ui_context_initializer.machine_mgr.add_machine(machine)
    machine.set_has_z_axis(False)
    assert not machine.has_z_axis
    dialog = MachineSettingsDialog(machine=machine)
    assert dialog._bed_mesh_stack_page is not None
    assert dialog._bed_mesh_row is not None
    dialog._update_bed_mesh_page_visibility()
    assert not dialog._bed_mesh_stack_page.get_visible()
    assert not dialog._bed_mesh_row.get_visible()

    machine.set_has_z_axis(True)
    dialog._update_bed_mesh_page_visibility()
    assert dialog._bed_mesh_stack_page.get_visible()
    assert dialog._bed_mesh_row.get_visible()
    dialog.destroy()


@pytest.mark.ui
def test_settings_dialog_selects_bed_mesh_initial_page(
    ui_context_initializer,
):
    machine = _make_machine()
    ui_context_initializer.machine_mgr.add_machine(machine)
    machine.set_has_z_axis(True)
    dialog = MachineSettingsDialog(machine=machine, initial_page="bed-mesh")
    selected = dialog.sidebar_list.get_selected_row()
    assert dialog._row_to_page_name[selected] == "bed-mesh"
    assert dialog._bed_mesh_row is not None
    assert dialog._bed_mesh_row.get_visible()
    dialog.destroy()


@pytest.mark.ui
def test_bed_mesh_page_apply_default_switch(ui_context_initializer):
    from rayforge.machine.models.bed_mesh import BedMesh

    machine = _make_machine()
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
    page = BedMeshPage(machine=machine)

    # Off by default; no mesh correction default configured.
    assert not page.apply_row.get_active()
    page.apply_row.set_active(True)
    page._on_apply_default_changed(page.apply_row, None)
    assert machine.default_post_processors_dicts == [
        {
            "name": "MeshCorrectionTransformer",
            "enabled": True,
            "z_offset": 0.0,
        }
    ]

    page.apply_row.set_active(False)
    page._on_apply_default_changed(page.apply_row, None)
    assert machine.default_post_processors_dicts == []
