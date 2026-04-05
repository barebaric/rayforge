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


class TestRenderConfig3D:
    @pytest.fixture
    def sample_config(self):
        w2v = np.eye(4, dtype=np.float32)
        w2v[0, 3] = -10.0
        w2v[1, 3] = -5.0

        w2c = np.eye(4, dtype=np.float32)
        w2c[2, 3] = 3.0

        cut_lut = np.zeros((256, 4), dtype=np.float32)
        cut_lut[:, 0] = np.linspace(0, 1, 256)
        cut_lut[:, 3] = 1.0

        engrave_lut = np.zeros((256, 4), dtype=np.float32)
        engrave_lut[:, 1] = np.linspace(0, 1, 256)
        engrave_lut[:, 3] = 1.0

        laser_luts = {
            "laser_a": {
                "cut": np.ones((256, 4), dtype=np.float32).tobytes(),
                "engrave": np.zeros((256, 4), dtype=np.float32).tobytes(),
            },
        }

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
            default_color_lut_cut=cut_lut.tobytes(),
            default_color_lut_engrave=engrave_lut.tobytes(),
            laser_color_luts=laser_luts,
            zero_power_rgba=(0.5, 0.5, 0.5, 1.0),
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
        assert sample_config.zero_power_rgba == restored.zero_power_rgba
        assert restored.layer_configs is not None
        assert len(restored.layer_configs) == 2
        assert restored.layer_configs["layer_0"].rotary_enabled is False
        assert restored.layer_configs["layer_1"].rotary_diameter == 50.0
        assert (
            restored.default_color_lut_cut
            == sample_config.default_color_lut_cut
        )
        assert (
            restored.laser_color_luts["laser_a"]["cut"]
            == sample_config.laser_color_luts["laser_a"]["cut"]
        )

    def test_round_trip_preserves_lut_bytes(self, sample_config):
        d = sample_config.to_dict()
        restored = RenderConfig3D.from_dict(d)
        orig_cut = np.frombuffer(
            sample_config.default_color_lut_cut, dtype=np.float32
        ).reshape(256, 4)
        restored_cut = np.frombuffer(
            restored.default_color_lut_cut, dtype=np.float32
        ).reshape(256, 4)
        np.testing.assert_allclose(orig_cut, restored_cut)

    def test_multiple_laser_luts(self):
        laser_luts = {
            "laser_1": {
                "cut": np.ones((256, 4), dtype=np.float32).tobytes(),
                "engrave": np.zeros((256, 4), dtype=np.float32).tobytes(),
            },
            "laser_2": {
                "cut": np.zeros((256, 4), dtype=np.float32).tobytes(),
                "engrave": np.ones((256, 4), dtype=np.float32).tobytes(),
            },
        }
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
            default_color_lut_cut=np.zeros(
                (256, 4), dtype=np.float32
            ).tobytes(),
            default_color_lut_engrave=np.zeros(
                (256, 4), dtype=np.float32
            ).tobytes(),
            laser_color_luts=laser_luts,
            zero_power_rgba=(1.0, 0.0, 0.0, 1.0),
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert set(restored.laser_color_luts.keys()) == {"laser_1", "laser_2"}

    def test_none_layer_configs(self):
        config = RenderConfig3D(
            world_to_visual=np.eye(4, dtype=np.float32),
            world_to_cyl_local=np.eye(4, dtype=np.float32),
            default_color_lut_cut=np.zeros(
                (256, 4), dtype=np.float32
            ).tobytes(),
            default_color_lut_engrave=np.zeros(
                (256, 4), dtype=np.float32
            ).tobytes(),
            laser_color_luts={},
            zero_power_rgba=(0.0, 0.0, 0.0, 0.0),
        )
        restored = RenderConfig3D.from_dict(config.to_dict())
        assert restored.layer_configs is None
        assert len(restored.laser_color_luts) == 0

    def test_missing_field_raises(self):
        with pytest.raises(KeyError):
            RenderConfig3D.from_dict({"world_to_visual": b"\x00" * 64})
