import pytest
import numpy as np

from rayforge.ui_gtk.sim3d.scene3d.render_config import (
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
            gear_ratio=2.0,
            reverse=True,
            axis_position_3d=(1.0, 2.0, 3.0),
            cylinder_dir=(0.0, 1.0, 0.0),
        )
        restored = LayerRenderConfig.from_dict(cfg.to_dict())
        assert restored.axis_position == 10.0
        assert restored.gear_ratio == 2.0
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

    def test_missing_field_raises(self):
        with pytest.raises(KeyError):
            RenderConfig3D.from_dict({"world_to_visual": b"\x00" * 64})
