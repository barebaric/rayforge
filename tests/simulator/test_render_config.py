import numpy as np
import pytest
from raygeo.compressed_array import CompressedArray

from rayforge.simulator.scene3d.render_config import (
    LayerRenderConfig,
    RenderConfig3D,
)


class TestLayerRenderConfig:
    def test_round_trip(self):
        cfg = LayerRenderConfig(
            rotary_enabled=True,
            rotary_diameter=50.0,
        )
        restored = LayerRenderConfig.from_dict(cfg.to_dict())
        assert restored.rotary_enabled is True
        assert restored.rotary_diameter == 50.0

    def test_missing_field_raises(self):
        with pytest.raises(KeyError):
            LayerRenderConfig.from_dict({"rotary_enabled": True})

    def test_optional_fields(self):
        cfg = LayerRenderConfig(
            rotary_enabled=False,
            rotary_diameter=25.0,
            axis_position=10.0,
            reverse=True,
            axis_position_3d=(1.0, 2.0, 3.0),
            cylinder_dir=(0.0, 1.0, 0.0),
        )
        restored = LayerRenderConfig.from_dict(cfg.to_dict())
        assert restored.axis_position == 10.0
        assert restored.reverse is True
        assert restored.axis_position_3d == (1.0, 2.0, 3.0)
        assert restored.cylinder_dir == (0.0, 1.0, 0.0)


class TestRenderConfig3D:
    @pytest.fixture
    def sample_config(self):
        w2v = np.eye(4, dtype=np.float32)
        w2v[0, 3] = -10.0
        w2v[1, 3] = -5.0

        w2c = np.eye(4, dtype=np.float32)
        w2c[2, 3] = 3.0

        return RenderConfig3D(
            world_to_visual=w2v,
            world_to_cyl_local=w2c,
            layer_configs={
                "layer_0": LayerRenderConfig(
                    rotary_enabled=False,
                    rotary_diameter=25.0,
                ),
                "layer_1": LayerRenderConfig(
                    rotary_enabled=True,
                    rotary_diameter=50.0,
                ),
            },
        )

    def test_round_trip(self, sample_config):
        d = sample_config.to_dict()
        restored = RenderConfig3D.from_dict(d)

        np.testing.assert_allclose(
            sample_config.world_to_visual, restored.world_to_visual
        )
        np.testing.assert_allclose(
            sample_config.world_to_cyl_local, restored.world_to_cyl_local
        )
        assert restored.layer_configs is not None
        assert len(restored.layer_configs) == 2
        assert restored.layer_configs["layer_0"].rotary_enabled is False
        assert restored.layer_configs["layer_1"].rotary_diameter == 50.0

    def test_round_trip_preserves_matrices(self, sample_config):
        d = sample_config.to_dict()
        restored = RenderConfig3D.from_dict(d)
        np.testing.assert_allclose(
            sample_config.world_to_visual, restored.world_to_visual
        )
        np.testing.assert_allclose(
            sample_config.world_to_cyl_local, restored.world_to_cyl_local
        )

    def test_none_layer_configs(self):
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.layer_configs is None

    def test_laser_dot_widths_round_trip(self):
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
            laser_dot_widths_mm={"head1": 0.1, "head2": 0.3},
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.laser_dot_widths_mm == {
            "head1": 0.1,
            "head2": 0.3,
        }

    def test_laser_dot_widths_none_when_absent(self):
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.laser_dot_widths_mm is None

    def test_missing_field_raises(self):
        with pytest.raises(KeyError):
            RenderConfig3D.from_dict({"world_to_visual": b"\x00" * 64})

    def test_stock_world_to_visual_round_trip(self):
        stock_w2v = np.eye(4, dtype=np.float32)
        stock_w2v[2, 3] = 0.0
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
            stock_world_to_visual=stock_w2v,
            stock_top_z=5.0,
            has_z_axis=False,
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.stock_world_to_visual is not None
        np.testing.assert_allclose(restored.stock_world_to_visual, stock_w2v)
        assert restored.stock_top_z == 5.0
        assert restored.has_z_axis is False

    def test_defaults_has_z_axis_true(self):
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.has_z_axis is True
        assert restored.stock_top_z == 0.0
        assert restored.stock_world_to_visual is None

    def test_stock_specs_burn_round_trip(self):
        """The burn entry (with its CompressedArray) passes through
        to_dict/from_dict unchanged — the config dict crosses the
        compile-thread boundary in-process."""
        burn = {
            "surface_map": CompressedArray.from_uint8_2d(
                np.full((4, 4), 255, dtype=np.uint8)
            ),
            "origin_mm": (1.0, 2.0),
            "px_per_mm": (10.0, 10.0),
            "size_px": (100, 50),
        }
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
            stock_specs=[{"name": "oak", "burn": burn}],
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.stock_specs is not None
        spec = restored.stock_specs[0]
        restored_burn = spec["burn"]
        np.testing.assert_array_equal(
            restored_burn["surface_map"].to_numpy(),
            burn["surface_map"].to_numpy(),
        )
        assert restored_burn["origin_mm"] == (1.0, 2.0)
        assert restored_burn["size_px"] == (100, 50)
