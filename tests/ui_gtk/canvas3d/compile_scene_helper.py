"""Shared test helper for scene compiler tests."""

import numpy as np

from rayforge.ui_gtk.canvas3d.render_config import (
    RenderConfig3D,
    StepRenderConfig,
)


def make_test_config():
    cut_lut = np.zeros((256, 4), dtype=np.float32)
    cut_lut[:, 0] = np.linspace(0, 1, 256)
    cut_lut[:, 3] = 1.0

    engrave_lut = np.zeros((256, 4), dtype=np.float32)
    engrave_lut[:, 1] = np.linspace(0, 1, 256)
    engrave_lut[:, 3] = 1.0

    return RenderConfig3D(
        world_to_visual=np.eye(4, dtype=np.float32),
        world_to_cyl_local=np.eye(4, dtype=np.float32),
        step_configs={
            "step1": StepRenderConfig(
                rotary_enabled=False,
                rotary_diameter=25.0,
                laser_uid="",
            ),
        },
        default_color_lut_cut=cut_lut.tobytes(),
        default_color_lut_engrave=engrave_lut.tobytes(),
        laser_color_luts={},
        zero_power_rgba=(0.5, 0.5, 0.5, 1.0),
    )
