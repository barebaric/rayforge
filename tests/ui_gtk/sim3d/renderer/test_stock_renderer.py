"""Tests for the PBR stock renderer and its off-thread preparation."""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.simulator.scene3d import CompiledSceneArtifact, StockLayer
from rayforge.ui_gtk.sim3d.renderer.stock_renderer import (
    StockRenderer,
    prepare_stock_layer,
)

# GL calls the renderer may issue while init/upload paths run.
_GL_VOID_CALLS = (
    "glBindTexture",
    "glTexParameteri",
    "glPixelStorei",
    "glTexImage2D",
    "glBindVertexArray",
    "glBindBuffer",
    "glBufferData",
    "glVertexAttribPointer",
    "glEnableVertexAttribArray",
    "glDrawArrays",
    "glDeleteTextures",
    "glDeleteVertexArrays",
    "glDeleteBuffers",
    "glActiveTexture",
    "glDisable",
    "glEnable",
    "glPolygonOffset",
    "glGenerateMipmap",
)

_GL_GEN_CALLS = (
    "glGenTextures",
    "glGenVertexArrays",
    "glGenBuffers",
)


def _make_layer(
    texture_path=None,
    transform=None,
    roughness=0.55,
    metallic=0.0,
    rgba=(1.0, 0.5, 0.25, 1.0),
):
    # Minimal prism: two triangles at z=0 / z=-10.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, -10.0],
            [1.0, 0.0, -10.0],
            [1.0, 1.0, -10.0],
        ],
        dtype=np.float32,
    )
    normals = np.tile([0.0, 0.0, 1.0], (6, 1)).astype(np.float32)
    uvs = np.tile([0.0, 0.0], (6, 1)).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
    return StockLayer(
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=indices,
        transform=(
            transform if transform is not None else np.eye(4, dtype=np.float32)
        ),
        texture_path=texture_path,
        texture_size_mm=300.0,
        roughness=roughness,
        metallic=metallic,
        fallback_rgba=rgba,
    )


@pytest.fixture
def renderer():
    r = StockRenderer()
    with ExitStack() as stack:
        for name in _GL_VOID_CALLS:
            stack.enter_context(patch(f"OpenGL.GL.{name}"))
        for name in _GL_GEN_CALLS:
            stack.enter_context(patch(f"OpenGL.GL.{name}", return_value=1))
        r.init_gl()
        yield r


# ── prepare_stock_layer ─────────────────────────────────────────


@pytest.mark.ui
def test_prepare_expands_indices():
    prepared = prepare_stock_layer(_make_layer())
    assert prepared.positions.shape == (6, 3)
    assert prepared.positions.dtype == np.float32
    assert prepared.normals.shape == (6, 3)
    assert prepared.uvs.shape == (6, 2)
    assert prepared.texture_key is None
    assert prepared.texture_pixels is None


@pytest.mark.ui
def test_prepare_carries_material_params():
    layer = _make_layer(roughness=0.2, metallic=0.9)
    prepared = prepare_stock_layer(layer)
    assert prepared.roughness == 0.2
    assert prepared.metallic == 0.9
    assert prepared.fallback_rgba == (1.0, 0.5, 0.25, 1.0)
    assert np.array_equal(prepared.transform, layer.transform)


@pytest.mark.ui
def test_prepare_decodes_existing_texture(tmp_path):
    tex = tmp_path / "oak.webp"
    tex.write_bytes(b"fake")
    stat = tex.stat()
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    key = (str(tex), stat.st_mtime_ns, stat.st_size)

    layer = _make_layer(texture_path=str(tex))
    with patch(
        "rayforge.ui_gtk.sim3d.renderer.stock_renderer._decode_texture_cached",
        return_value=(2, 2, pixels),
    ) as mock_decode:
        prepared = prepare_stock_layer(layer)

    mock_decode.assert_called_once()
    assert prepared.texture_key == key
    assert prepared.texture_pixels is pixels


@pytest.mark.ui
def test_prepare_missing_texture_falls_back(tmp_path):
    layer = _make_layer(texture_path=str(tmp_path / "missing.webp"))
    prepared = prepare_stock_layer(layer)
    assert prepared.texture_key is None
    assert prepared.texture_pixels is None


# ── StockRenderer ────────────────────────────────────────────────


@pytest.mark.ui
def test_upload_prepared_builds_instances(renderer):
    prepared = prepare_stock_layer(_make_layer())
    renderer.upload_prepared([prepared])
    assert len(renderer.instances) == 1
    instance = renderer.instances[0]
    assert instance["vertex_count"] == 6
    assert instance["roughness"] == 0.55
    assert instance["texture_id"] == 0


@pytest.mark.ui
def test_upload_caches_texture_by_key(renderer, tmp_path):
    tex = tmp_path / "oak.webp"
    tex.write_bytes(b"fake")
    stat = tex.stat()
    key = (str(tex), stat.st_mtime_ns, stat.st_size)
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)

    layer = _make_layer(texture_path=str(tex))
    with patch(
        "rayforge.ui_gtk.sim3d.renderer.stock_renderer._decode_texture_cached",
        return_value=(2, 2, pixels),
    ):
        prepared = prepare_stock_layer(layer)

    renderer.upload_prepared([prepared])
    first_id = renderer.instances[0]["texture_id"]
    assert first_id != 0
    assert renderer._texture_cache[key] == first_id

    # Re-uploading the same prepared layer reuses the GL texture.
    with patch.object(renderer, "_create_gl_texture") as mock_create:
        renderer.upload_prepared([prepared])
    mock_create.assert_not_called()
    assert renderer.instances[0]["texture_id"] == first_id


@pytest.mark.ui
def test_upload_evicts_unreferenced_textures(renderer, tmp_path):
    tex = tmp_path / "oak.webp"
    tex.write_bytes(b"fake")
    stat = tex.stat()
    key = (str(tex), stat.st_mtime_ns, stat.st_size)
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)

    layer = _make_layer(texture_path=str(tex))
    with patch(
        "rayforge.ui_gtk.sim3d.renderer.stock_renderer._decode_texture_cached",
        return_value=(2, 2, pixels),
    ):
        textured = prepare_stock_layer(layer)

    renderer.upload_prepared([textured])
    assert key in renderer._texture_cache

    plain = prepare_stock_layer(_make_layer())
    renderer.upload_prepared([plain])
    assert renderer._texture_cache == {}
    assert len(renderer.instances) == 1


@pytest.mark.ui
def test_update_from_artifact(renderer):
    artifact = CompiledSceneArtifact(
        generation_id=1,
        vertex_layers=[],
        overlay_layers=[],
        stock_layers=[_make_layer(), _make_layer()],
    )
    renderer.update_from_artifact(artifact)
    assert len(renderer.instances) == 2


@pytest.mark.ui
def test_render_skips_without_stock_shader(renderer):
    ctx = MagicMock()
    shaders = MagicMock()
    shaders.stock = None
    renderer.instances = [
        {
            "vao": 1,
            "vbos": [1],
            "vertex_count": 3,
            "transform": np.eye(4, dtype=np.float32),
            "roughness": 0.5,
            "metallic": 0.0,
            "fallback_rgba": (1, 1, 1, 1),
            "texture_id": 0,
        }
    ]
    with patch("OpenGL.GL.glDrawArrays") as mock_draw:
        renderer.render(ctx, shaders)
    mock_draw.assert_not_called()


@pytest.mark.ui
def test_clear_releases_instances(renderer):
    renderer.upload_prepared([prepare_stock_layer(_make_layer())])
    assert renderer.instances
    renderer.clear()
    assert renderer.instances == []
