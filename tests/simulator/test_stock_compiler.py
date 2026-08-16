"""Tests for CPU-side stock mesh compilation.

Geometry-precision coverage (triangulation area deviation, wall
winding, UV density) lives in raygeo's ``tests/mesh/test_mesh_prism.py``
against ``build_prism_mesh``; these tests cover the rayforge-side spec
extraction, validation and GPU buffer packing.
"""

import math

import numpy as np
import pytest

from rayforge.simulator.scene3d.compiled_scene import StockLayer
from rayforge.simulator.scene3d.stock_compiler import compile_stock_layers

RECT_OUTER = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]


def _identity():
    return np.eye(4, dtype=np.float32)


# ── Public API ──────────────────────────────────────────────────


def test_compile_stock_layers_empty():
    assert compile_stock_layers([], _identity()) == []


def test_compile_stock_layers_skips_degenerate():
    layers = compile_stock_layers([{"name": "bad", "outers": []}], _identity())
    assert layers == []


def test_compile_stock_layers_short_ring_skipped():
    spec = {"name": "sliver", "outers": [[(0.0, 0.0), (1.0, 1.0)]]}
    assert compile_stock_layers([spec], _identity()) == []


def test_compile_stock_layers_basic_spec():
    spec = {
        "name": "oak stock",
        "thickness": 18.0,
        "outers": [RECT_OUTER],
        "holes": [],
        "texture_path": "/tmp/oak.webp",
        "texture_size_mm": 250.0,
        "roughness": 0.55,
        "metallic": 0.0,
        "color": "#A0522D",
    }
    w2v = _identity()
    layers = compile_stock_layers([spec], w2v)
    assert len(layers) == 1
    layer = layers[0]
    assert isinstance(layer, StockLayer)
    assert layer.texture_path == "/tmp/oak.webp"
    assert layer.texture_size_mm == 250.0
    assert layer.roughness == 0.55
    assert layer.metallic == 0.0
    assert layer.fallback_rgba == pytest.approx(
        (0.6274509803921569, 0.3215686274509804, 0.17647058823529413, 1.0)
    )
    assert np.array_equal(layer.transform, w2v)
    assert layer.positions.dtype == np.float32
    assert layer.indices.dtype == np.uint32


def test_compile_stock_layers_rectangle_parity():
    spec = {
        "name": "rect",
        "thickness": 18.0,
        "outers": [RECT_OUTER],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    pos = layer.positions.reshape(-1, 3)
    norm = layer.normals.reshape(-1, 3)

    # 2 caps * 4 verts + 4 walls * 4 verts; flat triangle indices.
    assert pos.shape == (24, 3)
    assert layer.indices.shape == (36,)

    # z spans the bed (0) up to +thickness.
    assert pos[:, 2].max() == 18.0
    assert pos[:, 2].min() == 0.0

    # Every normal is unit length and walls are horizontal.
    assert np.allclose(np.linalg.norm(norm, axis=1), 1.0, atol=1e-6)
    assert np.allclose(norm[norm[:, 2] == 0.0, 2], 0.0)


def test_compile_stock_layers_multi_outer_offsets():
    spec = {
        "name": "two islands",
        "thickness": 10.0,
        "outers": [RECT_OUTER, RECT_OUTER],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    pos = layer.positions.reshape(-1, 3)
    idx = layer.indices

    # Both rings compiled back to back.
    assert pos.shape == (48, 3)
    assert idx.max() == 47
    # Second ring's indices are offset past the first ring's verts.
    assert int(idx.min()) >= 0
    assert (idx >= 24).any()


def test_compile_stock_layers_default_thickness():
    spec = {
        "name": "no thickness",
        "thickness": None,
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    assert layer.positions.reshape(-1, 3)[:, 2].max() == 18.0


def test_compile_stock_layers_invalid_thickness_falls_back():
    spec = {
        "name": "bad thickness",
        "thickness": "not-a-number",
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    assert layer.positions.reshape(-1, 3)[:, 2].max() == 18.0


def test_compile_stock_layers_invalid_color_falls_back():
    spec = {
        "name": "bad color",
        "thickness": 10.0,
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
        "color": "not-a-color",
    }
    layer = compile_stock_layers([spec], _identity())[0]
    assert layer.fallback_rgba == (1.0, 1.0, 1.0, 1.0)


def test_compile_stock_layers_bottom_on_bed():
    """Stock prism bottom is at z=0 (bed), top at z=+thickness."""
    spec = {
        "name": "oak",
        "thickness": 5.0,
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    pos = layer.positions.reshape(-1, 3)
    assert pos[:, 2].min() == pytest.approx(0.0)
    assert pos[:, 2].max() == pytest.approx(5.0)


def test_compile_stock_layers_uses_stock_world_to_visual():
    """Stock transform comes from stock_world_to_visual, not content w2v."""
    stock_w2v = np.eye(4, dtype=np.float32)
    stock_w2v[2, 3] = 0.0  # bed-anchored
    content_w2v = np.eye(4, dtype=np.float32)
    content_w2v[2, 3] = 10.0  # lifted for no-Z content

    spec = {
        "name": "oak",
        "thickness": 3.0,
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], stock_w2v)[0]
    # Transform must be the bed-anchored one, not the lifted content one.
    assert layer.transform[2, 3] == 0.0


# ── Rotary cylinder shells ───────────────────────────────────────


def _rotary_spec(**overrides):
    spec = {
        "name": "oak cylinder",
        "kind": "rotary",
        "diameter": 50.0,
        "length": 200.0,
        "texture_path": "/tmp/oak.webp",
        "texture_size_mm": 250.0,
        "roughness": 0.6,
        "metallic": 0.0,
        "color": "#A0522D",
    }
    spec.update(overrides)
    return spec


def test_compile_rotary_stock_basic_spec():
    layer = compile_stock_layers([_rotary_spec()], _identity())[0]
    assert layer.is_rotary is True
    assert layer.texture_path == "/tmp/oak.webp"
    assert layer.texture_size_mm == 250.0

    from rayforge.simulator.scene3d.stock_compiler import (
        CYLINDER_LENGTH_SEGMENTS,
        CYLINDER_RINGS,
    )

    pos = layer.positions.reshape(-1, 3)
    norm = layer.normals.reshape(-1, 3)
    shell_verts = (CYLINDER_LENGTH_SEGMENTS + 1) * (CYLINDER_RINGS + 1)
    assert pos.shape == (shell_verts + 2, 3)
    assert layer.indices.shape == (
        CYLINDER_LENGTH_SEGMENTS * CYLINDER_RINGS * 6 + CYLINDER_RINGS * 6,
    )

    # Axis along local X, spanning 0..length; radius = diameter/2.
    assert pos[:, 0].min() == pytest.approx(0.0)
    assert pos[:, 0].max() == pytest.approx(200.0)
    radius = np.hypot(pos[:, 1], pos[:, 2])
    assert radius[:shell_verts] == pytest.approx(25.0)
    # The two cap centers sit on the axis.
    assert np.all(radius[shell_verts:] == 0.0)
    assert pos[shell_verts, 0] == pytest.approx(0.0)
    assert pos[shell_verts + 1, 0] == pytest.approx(200.0)

    # Unit-length normals everywhere: radial on the shell, ±X caps.
    assert np.allclose(np.linalg.norm(norm, axis=1), 1.0, atol=1e-6)
    radial = (
        np.stack(
            [
                np.zeros(shell_verts),
                pos[:shell_verts, 1],
                pos[:shell_verts, 2],
            ],
            axis=-1,
        )
        / radius[:shell_verts, None]
    )
    assert np.allclose(norm[:shell_verts], radial, atol=1e-5)
    assert np.allclose(norm[shell_verts], (-1.0, 0.0, 0.0))
    assert np.allclose(norm[shell_verts + 1], (1.0, 0.0, 0.0))


def test_compile_rotary_stock_uv_physical_density():
    layer = compile_stock_layers(
        [_rotary_spec(texture_size_mm=100.0)], _identity()
    )[0]
    uvs = layer.uvs.reshape(-1, 2)
    # U follows the circumference: circumference / texture_size_mm
    # repeats, wrapping around the seam.
    circumference = 50.0 * math.pi
    assert uvs[:, 0].min() == pytest.approx(0.0)
    assert uvs[:, 0].max() == pytest.approx(circumference / 100.0)
    # V follows the axis: length / texture_size_mm repeats, so the
    # source texture's vertical grain runs along the cylinder.
    assert uvs[:, 1].min() == pytest.approx(0.0)
    assert uvs[:, 1].max() == pytest.approx(200.0 / 100.0)


def test_compile_rotary_stock_invalid_dimensions_skipped():
    assert (
        compile_stock_layers([_rotary_spec(diameter=0.0)], _identity()) == []
    )
    assert compile_stock_layers([_rotary_spec(length=-5.0)], _identity()) == []


def test_compile_rotary_stock_is_flat_fallback_without_kind():
    """Specs without 'kind' keep building flat prisms (legacy path)."""
    spec = {
        "name": "flat",
        "thickness": 4.0,
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    assert layer.is_rotary is False


def test_compile_stock_layers_preserves_panel_transform(
    lite_context, sync_machine
):
    """Stock inherits the panel presentation rotation from world_to_visual.

    The scene presenter builds world_to_visual as
    ``margin_shift @ world_to_panel`` so the stock prism lands in the
    same presented space as the ops, models and zones.
    """
    from rayforge.machine.models.machine_panel import PanelOrientation
    from rayforge.ui_gtk.sim3d.viewport import ViewportConfig

    sync_machine.set_axis_extents(400.0, 300.0)
    sync_machine.set_panel_orientation(PanelOrientation.ROTATED_RIGHT)
    vp = ViewportConfig.from_machine(sync_machine)

    w2v = np.identity(4, dtype=np.float32)
    w2v[:3, :] = (vp.margin_shift @ vp.world_to_panel)[:3, :]
    w2v[2, 3] = vp.wcs_offset_mm[2]

    spec = {
        "name": "oak",
        "thickness": 18.0,
        "outers": [
            [(100.0, 50.0), (120.0, 50.0), (120.0, 60.0), (100.0, 60.0)]
        ],
        "holes": [],
    }
    layer = compile_stock_layers([spec], w2v)[0]
    np.testing.assert_array_equal(layer.transform, w2v)

    corner = layer.transform @ np.array([100.0, 50.0, 0.0, 1.0])
    # ROTATED_RIGHT presents world (100, 50) at panel (50, 300).
    assert corner[:2] == pytest.approx([50.0, 300.0])
