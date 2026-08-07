"""
A composite renderer owning all scene GPU resources for the 3D canvas.

The SceneRenderer owns the child renderers and shaders plus the per-layer
collections (layer groups, cylinders, models).  Canvas3D is responsible for
the GL context lifecycle and per-frame state; it delegates resource creation,
rebuilds and theme/colour application to this class.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from ....simulator.scene3d import CompiledSceneArtifact
from ...shared.color_lut_provider import ColorLutProvider
from ..gl_state import render_pass
from ..gl_utils import LayerRenderer, RenderContext, ShaderSet
from ..layer_renderer_group import (
    LayerRendererGroup,
    match_overlay_layer,
    match_vertex_layer,
)
from ..shader import (
    BackgroundShader,
    Shader,
    SimpleShader,
    TextShader,
    TextureShader,
)
from ..viewport import ViewportConfig
from .axis_renderer_3d import AxisRenderer3D
from .background_renderer import BackgroundRenderer
from .base import BaseRenderer
from .cylinder_renderer import CylinderRenderer
from .laser_beam_renderer import LaserBeamRenderer
from .model_renderer import ModelRenderer
from .texture_renderer import TextureArtifactRenderer
from .zone_renderer import ZoneRenderer

logger = logging.getLogger(__name__)


class RingPassAdapter:
    """
    Draws a layer group's ring buffer as a separate registry pass.

    The layer group renders its toolpaths in the main pass; the ring
    buffer must draw *after* the texture renderer so scanlines appear on
    top of the texture quads.  This adapter exposes just the ring draw so
    the scene can place it late in the registry.
    """

    def __init__(self, group: LayerRendererGroup):
        self._group = group

    def prepare(self, ctx: RenderContext) -> None:
        pass

    def render(self, ctx: RenderContext, shaders: ShaderSet) -> None:
        self._group.render_ring(ctx, shaders)

    def init_gl(self) -> None:
        pass


class SceneRenderer(BaseRenderer):
    """Owns the GPU renderers, shaders and collections for the 3D scene."""

    def __init__(self):
        super().__init__()
        self.main_shader: Optional[Shader] = None
        self.text_shader: Optional[Shader] = None
        self.texture_shader: Optional[Shader] = None
        self.background_shader: Optional[Shader] = None
        self.shader_set: Optional[ShaderSet] = None

        self.axis_renderer: Optional[AxisRenderer3D] = None
        self.background_renderer: Optional[BackgroundRenderer] = (
            BackgroundRenderer()
        )
        self.texture_renderer: Optional[TextureArtifactRenderer] = None
        self.zone_renderer: Optional[ZoneRenderer] = None
        self.laser_beam_renderer: Optional[LaserBeamRenderer] = (
            LaserBeamRenderer()
        )

        self.layer_groups: List[LayerRendererGroup] = []
        self.cylinder_renderers: Dict[float, CylinderRenderer] = {}
        self.model_renderers: List[ModelRenderer] = []
        self.had_rotary_layers = False
        self.cylinder_transform = np.eye(4, dtype=np.float64)

        # Ordered list of (renderer, shader_keys) in draw order.  The
        # deferred ring passes come after the texture renderer so rings
        # draw on top of the textures during playback.
        self.render_registry: List[Tuple[LayerRenderer, Tuple[str, ...]]] = []

    def _rebuild_registry(self) -> None:
        """Rebuilds the render registry from the current children."""
        registry: List[Tuple[LayerRenderer, Tuple[str, ...]]] = []
        if self.background_renderer is not None:
            registry.append((self.background_renderer, ("background",)))
        if self.axis_renderer is not None:
            registry.append((self.axis_renderer, ("main", "text")))
        if self.zone_renderer is not None:
            registry.append((self.zone_renderer, ("main",)))
        for group in self.layer_groups:
            registry.append((group, ("main",)))
        for renderer in self.cylinder_renderers.values():
            registry.append((renderer, ("main",)))
        if self.texture_renderer is not None:
            registry.append((self.texture_renderer, ("texture",)))
        for group in self.layer_groups:
            registry.append((RingPassAdapter(group), ("main",)))
        if self.laser_beam_renderer is not None:
            registry.append((self.laser_beam_renderer, ("main",)))
        for renderer in self.model_renderers:
            registry.append((renderer, ("main",)))
        self.render_registry = registry

    def set_cylinder_transform(self, transform: np.ndarray):
        """Stores the assembly's cylinder base transform."""
        self.cylinder_transform = transform

    def init_gl(self, viewport: ViewportConfig, font_family: str):
        """Creates and initializes all scene shaders and renderers."""
        self.main_shader = SimpleShader()
        self.text_shader = TextShader()
        self.texture_shader = TextureShader()
        self.background_shader = BackgroundShader()
        self.shader_set = ShaderSet(
            main=self.main_shader,
            text=self.text_shader,
            texture=self.texture_shader,
            background=self.background_shader,
        )

        self.axis_renderer = AxisRenderer3D(
            viewport.width_mm,
            viewport.depth_mm,
            grid_size_mm=get_default_grid_step_mm(),
            grid_unit_factor=get_preferred_unit_factor("length"),
            font_family=font_family,
        )
        self.apply_extent_frame(viewport)
        self.axis_renderer.init_gl()
        self.texture_renderer = TextureArtifactRenderer()
        self.texture_renderer.init_gl()
        if self.laser_beam_renderer:
            self.laser_beam_renderer.init_gl()
        try:
            if self.background_renderer:
                self.background_renderer.init_gl()
        except Exception as e:
            logger.warning(
                "Background renderer init failed, "
                "falling back to clear color: %s",
                e,
            )
            self.background_renderer = None
        self.zone_renderer = ZoneRenderer()
        self.zone_renderer.init_gl()
        self._rebuild_registry()

    def _cleanup_self(self):
        """Cleans up all scene-owned renderers and shaders."""
        for group in self.layer_groups:
            group.cleanup()
        if self.axis_renderer:
            self.axis_renderer.cleanup()
        if self.laser_beam_renderer:
            self.laser_beam_renderer.cleanup()
        if self.background_renderer:
            self.background_renderer.cleanup()
        if self.texture_renderer:
            self.texture_renderer.cleanup()
        if self.zone_renderer:
            self.zone_renderer.cleanup()
        for renderer in self.cylinder_renderers.values():
            renderer.cleanup()
        for renderer in self.model_renderers:
            renderer.cleanup()
        if self.main_shader:
            self.main_shader.cleanup()
        if self.text_shader:
            self.text_shader.cleanup()
        if self.texture_shader:
            self.texture_shader.cleanup()
        if self.background_shader:
            self.background_shader.cleanup()

    def apply_extent_frame(self, viewport: ViewportConfig):
        """Applies the extent frame to the axis renderer if present."""
        if not self.axis_renderer or viewport.extent_frame is None:
            return
        fx, fy, fw, fh = viewport.extent_frame
        ml = -fx
        mb = -fy
        mt = fh - viewport.depth_mm - mb
        mr = fw - viewport.width_mm - ml
        if viewport.x_right:
            fx = -mr
        if viewport.y_down:
            fy = -mt
        self.axis_renderer.set_extent_frame(fx, fy, fw, fh, show=True)

    def update_axis_from_viewport(self, viewport: ViewportConfig) -> bool:
        """Rebuilds the axis renderer if the viewport dimensions changed."""
        if not self.axis_renderer:
            return False
        if (
            self.axis_renderer.width_mm == viewport.width_mm
            and self.axis_renderer.height_mm == viewport.depth_mm
        ):
            self.apply_extent_frame(viewport)
            return False
        font_family = self.axis_renderer.font_family
        self.axis_renderer.cleanup()
        self.axis_renderer = AxisRenderer3D(
            viewport.width_mm,
            viewport.depth_mm,
            grid_size_mm=get_default_grid_step_mm(),
            grid_unit_factor=get_preferred_unit_factor("length"),
            font_family=font_family,
        )
        self.apply_extent_frame(viewport)
        self.axis_renderer.init_gl()
        self._rebuild_registry()
        return True

    def update_cylinders_from_doc(self, doc, viewport, machine):
        """Reads chuck diameters from the assembly and rebuilds cylinders."""
        desired_diameters: Dict[float, bool] = {}
        if machine and self.had_rotary_layers:
            for layer in doc.layers:
                if layer.rotary_enabled and layer.rotary_diameter > 0:
                    desired_diameters[layer.rotary_diameter] = True

        max_length = viewport.width_mm
        if machine:
            default_rm = machine.get_default_rotary_module()
            if default_rm:
                max_length = min(max_length, default_rm.max_workpiece_length)

        for diameter, renderer in list(self.cylinder_renderers.items()):
            if diameter not in desired_diameters:
                renderer.cleanup()
                del self.cylinder_renderers[diameter]

        grid_size = (
            self.axis_renderer.grid_size_mm if self.axis_renderer else 10.0
        )
        length_segments = max(1, round(max_length / grid_size))

        for diameter in desired_diameters:
            if diameter not in self.cylinder_renderers:
                renderer = CylinderRenderer(
                    diameter=diameter,
                    length=max_length,
                    rings=24,
                    length_segments=length_segments,
                )
                renderer.set_color((0.4, 0.6, 0.8, 0.25))
                renderer.init_gl()
                self.cylinder_renderers[diameter] = renderer
                logger.debug(
                    f"Initialized cylinder renderer: "
                    f"diameter={diameter}mm, length={max_length}mm"
                )
        self._rebuild_registry()

    def update_zones_from_machine(self, machine):
        """Pushes the machine's no-go zones into the zone renderer."""
        if not self.zone_renderer:
            return
        if not machine:
            return
        zones = list(machine.nogo_zones.values())
        self.zone_renderer.update_zones(zones)

    def clear_models(self):
        """Removes all model renderers without rebuilding."""
        for renderer in self.model_renderers:
            renderer.cleanup()
        self.model_renderers.clear()
        self._rebuild_registry()

    def update_models_from_context(self, context, machine):
        """Rebuilds renderers for all assembly links with 3D models."""
        self.clear_models()
        if not machine:
            return

        assembly = machine.assembly
        if assembly is None:
            return

        model_links = assembly.get_model_links()
        logger.debug("Model renderers: %d links with models", len(model_links))

        for link in model_links:
            assert link.model is not None
            logger.debug(
                "Model renderers: resolving model %s for link %s",
                link.model,
                link.name,
            )
            resolved = context.model_mgr.resolve(link.model)
            if resolved is None:
                logger.warning(
                    "Model file not found: %s, skipping.",
                    link.model.path,
                )
                continue

            renderer = ModelRenderer(resolved, link_name=link.name)
            renderer.init_gl()
            logger.debug(
                "Model renderer created: vao=%d, vertex_count=%d, bounds=%s",
                renderer._vao,
                renderer._vertex_count,
                renderer.bounds,
            )
            self.model_renderers.append(renderer)
        self._rebuild_registry()

    def apply_background_colors(self, bg_color, bg_light):
        """Applies the resolved background colors to the background."""
        if self.background_renderer:
            self.background_renderer.set_colors(bg_color, bg_light)

    def apply_axis_colors(self, axis_color, grid_color, bg_plane_color):
        """Applies the resolved foreground colors to the axis renderer."""
        if self.axis_renderer:
            self.axis_renderer.set_background_color(bg_plane_color)
            self.axis_renderer.set_axis_color(axis_color)
            self.axis_renderer.set_label_color(axis_color)
            self.axis_renderer.set_grid_color(grid_color)

    def update_color_luts(self, provider: Optional[ColorLutProvider]):
        """Fans out the shared colour LUT provider to all consumers."""
        if provider is None:
            return
        for group in self.layer_groups:
            group.ops_renderer.update_color_lut_from(provider)
            group.ring_renderer.update_color_lut_from(provider)

        if self.texture_renderer:
            if provider.has_lasers:
                logger.debug(
                    f"[COLOR_LUT] Using multi-laser 2D LUT "
                    f"({provider.num_lasers} lasers)"
                )
            self.texture_renderer.update_color_lut_from(provider)

    def update_from_artifact(
        self, artifact: CompiledSceneArtifact, show_travel_moves: bool
    ):
        """Rebuilds layer groups/textures synchronously from an artifact."""
        for group in self.layer_groups:
            group.cleanup()
        self.layer_groups.clear()

        for vl in artifact.vertex_layers:
            ol = match_overlay_layer(artifact.overlay_layers, vl.is_rotary)
            group = LayerRendererGroup(is_rotary=vl.is_rotary)
            group.init_gl()
            group.update_from_artifact(vl, ol, show_travel_moves)
            self.layer_groups.append(group)

        if self.texture_renderer:
            self.texture_renderer.update_from_artifact(artifact)
        self._rebuild_registry()

    def prepare_chunked_upload(
        self, artifact: CompiledSceneArtifact, show_travel_moves: bool
    ) -> List:
        """Creates fresh layer groups and returns the deferred upload items."""
        for group in self.layer_groups:
            group.cleanup()
        self.layer_groups.clear()

        upload_items: List[Any] = [("color_luts",)]

        for vl in artifact.vertex_layers:
            group = LayerRendererGroup(is_rotary=vl.is_rotary)
            group.init_gl()
            self.layer_groups.append(group)
            upload_items.append(("ops", group, vl, show_travel_moves))

        for ol in artifact.overlay_layers:
            for group in self.layer_groups:
                if group.is_rotary == ol.is_rotary:
                    upload_items.append(("overlay", group, ol))
                    break

        upload_items.append(("textures", artifact))
        upload_items.append(("op_player",))
        self._rebuild_registry()
        return upload_items

    def extract_playback_offsets(self, artifact: CompiledSceneArtifact):
        """Stores each layer group's playback offsets from an artifact."""
        for group in self.layer_groups:
            vl = match_vertex_layer(artifact.vertex_layers, group.is_rotary)
            if vl is not None:
                group.powered_offsets = vl.powered_cmd_offsets
                group.travel_offsets = vl.travel_cmd_offsets
            else:
                group.powered_offsets = []
                group.travel_offsets = []

            ol = match_overlay_layer(artifact.overlay_layers, group.is_rotary)
            if ol is not None:
                group.ring_offsets = ol.cmd_offsets
            else:
                group.ring_offsets = []

    def render(self, ctx: RenderContext) -> None:
        """
        Renders the whole scene for one frame via the render registry.

        All per-frame state is read from ``ctx`` (populated by the
        caller).  Every renderer is prepared before any render so that
        frame-level cross-dependencies (e.g. the laser point light
        feeding the model renderers) resolve before any draw.
        """
        for shader in (
            self.main_shader,
            self.text_shader,
            self.texture_shader,
            self.background_shader,
        ):
            if shader:
                shader.reset_uniforms()

        shaders = self.shader_set
        if shaders is None:
            return

        for renderer, _ in self.render_registry:
            renderer.prepare(ctx)

        for renderer, shader_keys in self.render_registry:
            pass_shaders = tuple(getattr(shaders, key) for key in shader_keys)
            with render_pass(*pass_shaders):
                renderer.render(ctx, shaders)
