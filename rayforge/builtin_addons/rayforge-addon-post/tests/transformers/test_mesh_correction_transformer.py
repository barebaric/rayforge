from typing import cast

import pytest
from helpers import FakeMachine
from post_processors.transformers import MeshCorrectionTransformer

from rayforge.machine.models.machine import Machine


@pytest.fixture
def transformer() -> MeshCorrectionTransformer:
    """Provides a default, enabled MeshCorrectionTransformer instance."""
    return MeshCorrectionTransformer(enabled=True)


@pytest.fixture
def machine_with_mesh() -> FakeMachine:
    from rayforge.machine.models.bed_mesh import BedMesh

    mesh = BedMesh()
    mesh.set_params(
        x0=0.0,
        y0=0.0,
        dx=100.0,
        dy=100.0,
        nx=2,
        ny=2,
        heights=[0.0, 1.0, 2.0, 3.0],
    )
    return FakeMachine(bed_mesh=mesh)


def test_serialization_round_trip():
    original = MeshCorrectionTransformer(enabled=False)
    data = original.to_dict()
    assert data["name"] == "MeshCorrectionTransformer"
    assert data["enabled"] is False

    recreated = MeshCorrectionTransformer.from_dict(data)
    assert isinstance(recreated, MeshCorrectionTransformer)
    assert recreated.enabled is False


def test_to_spec_raises_without_mesh(
    transformer: MeshCorrectionTransformer,
):
    with pytest.raises(ValueError, match="no probed bed mesh"):
        transformer.to_spec(None, None, None)
    with pytest.raises(ValueError, match="no probed bed mesh"):
        transformer.to_spec(None, None, cast(Machine, FakeMachine()))


def test_to_spec_builds_rust_spec(
    transformer: MeshCorrectionTransformer, machine_with_mesh
):
    spec = transformer.to_spec(None, None, cast(Machine, machine_with_mesh))
    assert spec.x0 == 0.0
    assert spec.dx == 100.0
    assert spec.heights.shape == (2, 2)


def test_applies_mesh_to_ops(
    transformer: MeshCorrectionTransformer, machine_with_mesh
):
    from raygeo.ops import Ops

    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(100.0, 100.0, 1.0)

    specs = [transformer.to_spec(None, None, cast(Machine, machine_with_mesh))]
    Ops.apply_transformers(ops, specs, progress_cb=None)

    assert ops.endpoint(0) == (0.0, 0.0, 0.0)
    # Grid corner (100, 100) has height 3.0; added to the original z=1.
    assert ops.endpoint(1) == (100.0, 100.0, 4.0)


def test_interpolation_midpoint_is_corner_average(
    transformer: MeshCorrectionTransformer, machine_with_mesh
):
    from raygeo.ops import Ops

    ops = Ops()
    ops.move_to(50.0, 50.0, 0.0)

    specs = [transformer.to_spec(None, None, cast(Machine, machine_with_mesh))]
    Ops.apply_transformers(ops, specs, progress_cb=None)
    assert ops.endpoint(0)[2] == pytest.approx(1.5)
