"""Smoketests for the SceneRenderer composite."""

from unittest.mock import MagicMock, patch

from rayforge.ui_gtk.sim3d.renderer.background_renderer import (
    BackgroundRenderer,
)
from rayforge.ui_gtk.sim3d.renderer.laser_beam_renderer import (
    LaserBeamRenderer,
)
from rayforge.ui_gtk.sim3d.renderer.scene_renderer import SceneRenderer


def test_scene_renderer_constructs_children():
    scene = SceneRenderer()

    assert isinstance(scene.laser_beam_renderer, LaserBeamRenderer)
    assert isinstance(scene.background_renderer, BackgroundRenderer)
    assert scene.layer_groups == []
    assert scene.cylinder_renderers == {}
    assert scene.model_renderers == []
    assert scene.had_rotary_layers is False
    assert scene.axis_renderer is None
    assert scene.texture_renderer is None
    assert scene.zone_renderer is None
    assert scene.main_shader is None
    assert scene.text_shader is None
    assert scene.texture_shader is None


def test_scene_renderer_init_gl_creates_children():
    scene = SceneRenderer()
    viewport = MagicMock()
    viewport.width_mm = 100.0
    viewport.depth_mm = 100.0
    viewport.extent_frame = None

    with (
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.SimpleShader"
        ) as mock_simple,
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.TextShader"
        ) as mock_text,
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.TextureShader"
        ) as mock_texture,
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.AxisRenderer3D"
        ) as mock_axis,
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer."
            "TextureArtifactRenderer"
        ) as mock_tex,
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.ZoneRenderer"
        ) as mock_zone,
        patch.object(LaserBeamRenderer, "init_gl"),
        patch.object(BackgroundRenderer, "init_gl"),
    ):
        scene.init_gl(viewport, "sans-serif")

    assert scene.axis_renderer is mock_axis.return_value
    assert scene.texture_renderer is mock_tex.return_value
    assert scene.zone_renderer is mock_zone.return_value
    mock_axis.return_value.init_gl.assert_called_once()
    mock_tex.return_value.init_gl.assert_called_once()
    mock_zone.return_value.init_gl.assert_called_once()
    mock_simple.assert_called_once()
    mock_text.assert_called_once()
    mock_texture.assert_called_once()
