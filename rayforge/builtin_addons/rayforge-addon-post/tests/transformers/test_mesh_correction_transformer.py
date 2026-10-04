import numpy as np
import pytest
from post_processors.transformers import MeshCorrectionTransformer


@pytest.fixture
def transformer() -> MeshCorrectionTransformer:
    """Provides a default, enabled MeshCorrectionTransformer instance."""
    return MeshCorrectionTransformer(enabled=True)


@pytest.fixture
def bed_mesh_settings() -> dict:
    """A 2x2 bed mesh covering 0..100 mm in both axes."""
    return {
        "bed_mesh": {
            "x0": 0.0,
            "y0": 0.0,
            "dx": 100.0,
            "dy": 100.0,
            "nx": 2,
            "ny": 2,
            "heights": [0.0, 1.0, 2.0, 3.0],
        }
    }


def test_serialization_round_trip():
    original = MeshCorrectionTransformer(enabled=False, z_offset=1.5)
    data = original.to_dict()
    assert data["name"] == "MeshCorrectionTransformer"
    assert data["enabled"] is False
    assert data["z_offset"] == 1.5

    recreated = MeshCorrectionTransformer.from_dict(data)
    assert isinstance(recreated, MeshCorrectionTransformer)
    assert recreated.enabled is False
    assert recreated.z_offset == 1.5


def test_to_spec_raises_without_mesh(
    transformer: MeshCorrectionTransformer,
):
    with pytest.raises(ValueError, match="no probed bed mesh"):
        transformer.to_spec(None, None, settings=None)
    with pytest.raises(ValueError, match="no probed bed mesh"):
        transformer.to_spec(None, None, settings={})


def test_to_spec_builds_rust_spec(
    transformer: MeshCorrectionTransformer, bed_mesh_settings
):
    spec = transformer.to_spec(None, None, bed_mesh_settings)
    assert spec.x0 == 0.0
    assert spec.dx == 100.0
    assert spec.z_offset == 0.0
    assert spec.heights.shape == (2, 2)


def test_to_spec_invalid_mesh_raises(
    transformer: MeshCorrectionTransformer,
):
    with pytest.raises(ValueError, match="dx"):
        transformer.to_spec(
            None,
            None,
            {
                "bed_mesh": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "dx": 0.0,
                    "dy": 10.0,
                    "nx": 2,
                    "ny": 2,
                    "heights": [0.0] * 4,
                }
            },
        )


def test_applies_mesh_to_ops(
    transformer: MeshCorrectionTransformer, bed_mesh_settings
):
    from raygeo.ops import Ops

    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(100.0, 100.0, 1.0)

    specs = [transformer.to_spec(None, None, bed_mesh_settings)]
    Ops.apply_transformers(ops, specs, progress_cb=None)

    assert ops.endpoint(0) == (0.0, 0.0, 0.0)
    # Grid corner (100, 100) has height 3.0; added to the original z=1.
    assert ops.endpoint(1) == (100.0, 100.0, 4.0)


def test_z_offset_added(
    bed_mesh_settings,
):
    from raygeo.ops import Ops

    transformer = MeshCorrectionTransformer(enabled=True, z_offset=-0.5)
    ops = Ops()
    ops.move_to(0.0, 0.0, 1.0)

    specs = [transformer.to_spec(None, None, bed_mesh_settings)]
    Ops.apply_transformers(ops, specs, progress_cb=None)

    assert ops.endpoint(0) == (0.0, 0.0, 0.5)


def test_numpy_mesh_produces_correct_interpolation(
    transformer: MeshCorrectionTransformer,
):
    """Interpolation midpoint of the 2x2 grid is the corner average."""
    from raygeo.ops import Ops

    ops = Ops()
    ops.move_to(50.0, 50.0, 0.0)
    settings = {
        "bed_mesh": {
            "x0": 0.0,
            "y0": 0.0,
            "dx": 100.0,
            "dy": 100.0,
            "nx": 2,
            "ny": 2,
            "heights": [0.0, 2.0, 4.0, 6.0],
        }
    }
    assert np.allclose(
        transformer.to_spec(None, None, settings).heights,
        np.array([[0.0, 2.0], [4.0, 6.0]]),
    )

    specs = [transformer.to_spec(None, None, settings)]
    Ops.apply_transformers(ops, specs, progress_cb=None)
    assert ops.endpoint(0)[2] == pytest.approx(3.0)
