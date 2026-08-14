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
    assert scene.ops_renderers == []
    assert scene.ring_renderers == []
    assert scene.cylinder_renderers == {}
    assert scene.model_renderers == []
    assert scene.had_rotary_layers is False
    assert scene.axis_renderer is None
    assert scene.texture_renderer is None
    assert scene.zone_renderer is None
    assert scene.main_shader is None
    assert scene.text_shader is None
    assert scene.texture_shader is None
    assert scene.background_shader is None
    assert scene.shader_set is None
    assert scene.render_registry == []


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
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.BackgroundShader"
        ) as mock_background,
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
        scene.set_viewport(viewport)
        scene.set_font_family("sans-serif")
        scene.init_gl()

    assert scene.axis_renderer is mock_axis.return_value
    assert scene.texture_renderer is mock_tex.return_value
    assert scene.zone_renderer is mock_zone.return_value
    assert scene.background_shader is mock_background.return_value
    assert scene.shader_set is not None
    assert scene.shader_set.main is mock_simple.return_value
    assert scene.shader_set.text is mock_text.return_value
    assert scene.shader_set.texture is mock_texture.return_value
    assert scene.shader_set.background is mock_background.return_value
    mock_axis.return_value.init_gl.assert_called_once()
    mock_tex.return_value.init_gl.assert_called_once()
    mock_zone.return_value.init_gl.assert_called_once()
    mock_simple.assert_called_once()
    mock_text.assert_called_once()
    mock_texture.assert_called_once()
    mock_background.assert_called_once()

    renderers = [r for r, _ in scene.render_registry]
    assert renderers == [
        scene.background_renderer,
        scene.axis_renderer,
        scene.zone_renderer,
        scene.texture_renderer,
        scene.laser_beam_renderer,
    ]


def test_cleanup_walks_static_children_once():
    """cleanup() cleans registered static children via the base walk."""
    scene = SceneRenderer()
    static_children = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    for child in static_children:
        scene._add_child_renderer(child)

    scene.cleanup()

    for child in static_children:
        child.cleanup.assert_called_once()


def test_update_axis_from_viewport_swaps_child_registration():
    """Rebuilding the axis renderer replaces it in the child registry."""
    scene = SceneRenderer()
    old_axis = MagicMock()
    new_axis = MagicMock()
    old_axis.width_mm = 100.0
    old_axis.height_mm = 100.0
    old_axis.font_family = "sans-serif"
    scene.axis_renderer = old_axis
    scene._add_child_renderer(old_axis)

    viewport = MagicMock()
    viewport.width_mm = 200.0
    viewport.depth_mm = 200.0
    viewport.extent_frame = None

    with (
        patch(
            "rayforge.ui_gtk.sim3d.renderer.scene_renderer.AxisRenderer3D"
        ) as mock_axis,
        patch.object(new_axis, "init_gl"),
        patch.object(scene, "apply_extent_frame"),
    ):
        mock_axis.return_value = new_axis
        result = scene.update_axis_from_viewport(viewport)

    assert result is True
    old_axis.cleanup.assert_called_once()
    assert old_axis not in scene._owned_renderers
    assert new_axis in scene._owned_renderers


def test_render_registry_orders_rings_after_texture():
    scene = SceneRenderer()

    ops = MagicMock()
    ring = MagicMock()
    scene.ops_renderers = [ops]
    scene.ring_renderers = [ring]
    scene.cylinder_renderers = {25.0: MagicMock()}
    scene.model_renderers = [MagicMock()]
    scene.texture_renderer = MagicMock()

    scene._rebuild_registry()

    renderers = [r for r, _ in scene.render_registry]
    assert ring in renderers
    assert ops in renderers
    ring_index = renderers.index(ring)
    texture_index = renderers.index(scene.texture_renderer)
    ops_index = renderers.index(ops)
    # The ring trail always draws on top of the toolpath and the raster.
    assert ring_index > texture_index
    # The toolpath draws above the raster texture but below the trail.
    assert ops_index > texture_index
    assert ops_index < ring_index


def test_prepare_runs_laser_beam_before_models():
    """The laser beam publishes the point-light position that the model
    renderers consume, so its prepare phase must run first even though
    it draws last."""
    scene = SceneRenderer()
    laser = MagicMock()
    scene.laser_beam_renderer = laser
    model = MagicMock()
    scene.model_renderers = [model]
    scene.ring_renderers = [MagicMock()]
    scene._rebuild_registry()

    prepare_order = []
    laser.prepare.side_effect = lambda *a, **k: prepare_order.append("laser")
    model.prepare.side_effect = lambda *a, **k: prepare_order.append("model")

    scene.prepare(MagicMock())

    assert prepare_order == ["laser", "model"]
