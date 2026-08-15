"""Tests for CPU-side stock mesh compilation.

Geometry-precision coverage (triangulation area deviation, wall
winding, UV density) lives in raygeo's ``tests/mesh/test_mesh_prism.py``
against ``build_prism_mesh``; these tests cover the rayforge-side spec
extraction, validation and GPU buffer packing.
"""

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

    # z spans the engrave plane down to -thickness.
    assert pos[:, 2].max() == 0.0
    assert pos[:, 2].min() == -18.0

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
    assert layer.positions.reshape(-1, 3)[:, 2].min() == -18.0


def test_compile_stock_layers_invalid_thickness_falls_back():
    spec = {
        "name": "bad thickness",
        "thickness": "not-a-number",
        "outers": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]],
        "holes": [],
    }
    layer = compile_stock_layers([spec], _identity())[0]
    assert layer.positions.reshape(-1, 3)[:, 2].min() == -18.0


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
