"""Shared test helper for scene compiler tests."""

import numpy as np

from rayforge.ui_gtk.sim3d.scene3d.render_config import (
    LayerRenderConfig,
    RenderConfig3D,
)


def make_test_config(
    layer_configs=None,
):
    cut_lut = np.zeros((256, 4), dtype=np.float32)
    cut_lut[:, 0] = np.linspace(0, 1, 256)
    cut_lut[:, 3] = 1.0

    engrave_lut = np.zeros((256, 4), dtype=np.float32)
    engrave_lut[:, 1] = np.linspace(0, 1, 256)
    engrave_lut[:, 3] = 1.0

    return RenderConfig3D(
        world_to_visual=np.eye(4, dtype=np.float32),
        world_to_cyl_local=np.eye(4, dtype=np.float32),
        default_color_lut_cut=cut_lut.tobytes(),
        default_color_lut_engrave=engrave_lut.tobytes(),
        laser_color_luts={},
        zero_power_rgba=(0.5, 0.5, 0.5, 1.0),
        layer_configs=layer_configs,
    )


def make_flat_layer_config():
    return LayerRenderConfig(
        rotary_enabled=False,
        rotary_diameter=0.0,
    )


def make_rotary_layer_config(diameter=50.0):
    return LayerRenderConfig(
        rotary_enabled=True,
        rotary_diameter=diameter,
    )


def make_assembled_ops(step_ops_list):
    """Assemble per-step ops into full job ops with markers.

    step_ops_list: list of (layer_uid, ops) tuples.
    Returns a single Ops with JobStart/End and LayerStart/End markers.
    """
    from rayforge.core.ops import Ops

    assembled = Ops()
    assembled.job_start()
    for layer_uid, step_ops in step_ops_list:
        assembled.layer_start(layer_uid)
        assembled.extend(step_ops)
        assembled.layer_end(layer_uid)
    assembled.job_end()
    return assembled
