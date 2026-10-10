"""Tests for the BedMesh model and its Machine/profile integration."""

import pytest

from rayforge.machine.models.bed_mesh import BedMesh
from rayforge.machine.models.machine import Machine


def _make_mesh() -> BedMesh:
    mesh = BedMesh()
    mesh.set_params(
        x0=10.0,
        y0=20.0,
        dx=25.0,
        dy=25.0,
        nx=3,
        ny=2,
        heights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        probe_params={"feed_mm_min": 120.0},
    )
    return mesh


class TestBedMesh:
    def test_sample_at_grid_nodes(self):
        mesh = _make_mesh()
        assert mesh.sample(10.0, 20.0) == pytest.approx(1.0)
        assert mesh.sample(60.0, 20.0) == pytest.approx(3.0)
        assert mesh.sample(10.0, 45.0) == pytest.approx(4.0)

    def test_sample_bilinear_midpoint(self):
        mesh = _make_mesh()
        assert mesh.sample(35.0, 32.5) == pytest.approx(3.5)

    def test_sample_clamps_outside_grid(self):
        mesh = _make_mesh()
        assert mesh.sample(-100.0, -100.0) == pytest.approx(1.0)
        assert mesh.sample(1000.0, 1000.0) == pytest.approx(6.0)

    def test_stats(self):
        mesh = _make_mesh()
        assert mesh.mean_z == pytest.approx(3.5)
        assert mesh.z_range == (1.0, 6.0)

    def test_set_params_rejects_size_mismatch(self):
        mesh = BedMesh()
        with pytest.raises(ValueError):
            mesh.set_params(
                x0=0, y0=0, dx=1, dy=1, nx=2, ny=2, heights=[1.0, 2.0]
            )

    def test_set_params_rejects_bad_grid(self):
        mesh = BedMesh()
        with pytest.raises(ValueError):
            mesh.set_params(x0=0, y0=0, dx=1, dy=1, nx=0, ny=0, heights=[])

    def test_round_trip(self):
        mesh = _make_mesh()
        restored = BedMesh.from_dict(mesh.to_dict())
        assert restored.x0 == mesh.x0
        assert restored.dx == mesh.dx
        assert restored.nx == mesh.nx
        assert restored.ny == mesh.ny
        assert restored.heights == mesh.heights
        assert restored.probe_params == mesh.probe_params
        assert restored.sample(35.0, 32.5) == pytest.approx(3.5)

    def test_from_dict_malformed_heights_falls_back(self):
        data = _make_mesh().to_dict()
        data["heights"] = [1.0, 2.0]  # wrong length for 3x2
        restored = BedMesh.from_dict(data)
        assert restored.nx == 1
        assert restored.ny == 1
        assert len(restored.heights) == 1

    def test_probed_at_timestamped(self):
        mesh = _make_mesh()
        assert mesh.probed_at


class TestMachineBedMesh:
    def _machine(self) -> Machine:
        from rayforge.context import RayforgeContext

        return Machine(RayforgeContext())

    def test_default_is_none(self):
        machine = self._machine()
        assert machine.bed_mesh is None

    def test_set_and_clear_sends_changed(self):
        machine = self._machine()
        events = []

        def on_changed(*args):
            events.append(args)

        machine.changed.connect(on_changed)

        mesh = _make_mesh()
        machine.set_bed_mesh(mesh)
        assert machine.bed_mesh is mesh
        assert len(events) == 1

        machine.clear_bed_mesh()
        assert machine.bed_mesh is None
        assert len(events) == 2

        # Clearing without a mesh is a no-op.
        machine.clear_bed_mesh()
        assert len(events) == 2

    def test_to_dict_from_dict_round_trip(self):
        machine = self._machine()
        machine.set_bed_mesh(_make_mesh())

        data = machine.to_dict()
        assert data["machine"]["bed_mesh"]["nx"] == 3

        restored = Machine.from_dict(data)
        mesh = restored.bed_mesh
        assert mesh is not None
        assert mesh.sample(35.0, 32.5) == pytest.approx(3.5)
