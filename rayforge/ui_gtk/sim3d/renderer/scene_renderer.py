"""
A composite renderer owning all scene GPU resources for the 3D canvas.

The SceneRenderer owns the child renderers and shaders plus the per-layer
collections (ops/ring renderers, cylinders, models).  Canvas3D is
responsible for the GL context lifecycle and per-frame state; it delegates
resource creation, rebuilds and theme/colour application to this class.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from OpenGL.error import GLError

from ....shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from ....simulator.scene3d import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    VertexLayer,
)
from ...shared.color_lut_provider import ColorLutProvider
from ..gl_state import render_pass
from ..gl_utils import ShaderSet
from ..render_context import RenderContext
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
from .ops_renderer import OpsRenderer, OpsUploadPayload, prepare_vertex_layer
from .ring_buffer_renderer import RingBufferRenderer
from .texture_renderer import TextureArtifactRenderer
from .zone_renderer import ZoneRenderer

logger = logging.getLogger(__name__)


def match_vertex_layer(
    vertex_layers: list[VertexLayer], is_rotary: bool
) -> VertexLayer | None:
    """Returns the vertex layer matching the given rotary flag."""
    for vl in vertex_layers:
        if vl.is_rotary == is_rotary:
            return vl
    return None


def match_overlay_layer(
    overlay_layers: list[ScanlineOverlayLayer], is_rotary: bool
) -> ScanlineOverlayLayer | None:
    """Returns the overlay layer matching the given rotary flag."""
    for ol in overlay_layers:
        if ol.is_rotary == is_rotary:
            return ol
    return None


class UploadItem(Protocol):
    """A single unit of work in a chunked scene upload."""

    def prepare(self) -> "OpsUploadPayload | None":
        """Prepares this item's data off the main thread."""
        raise NotImplementedError

    def upload(self) -> None:
        """Uploads this item's data into its renderer."""
        raise NotImplementedError


@dataclass
class OpsLayerUploadItem:
    """Uploads a vertex layer into an ops renderer."""

    renderer: "OpsRenderer"
    vertex_layer: VertexLayer
    show_travel_moves: bool
    _payload: OpsUploadPayload | None = field(
        default=None, init=False, repr=False
    )

    def prepare(self) -> OpsUploadPayload:
        """Decompresses/concat vertex arrays off the main thread.

        Stores the built payload on the item so the main-thread
        ``upload`` only performs the GL buffer uploads.
        """
        payload = prepare_vertex_layer(
            self.vertex_layer, self.show_travel_moves
        )
        self._payload = payload
        return payload

    def upload(self) -> None:
        payload = self._payload
        if payload is None:
            payload = self.prepare()
        self.renderer.update_from_vertex_data(
            payload.powered_vertices,
            payload.powered_attrib,
            payload.travel_vertices,
        )


@dataclass
class OverlayLayerUploadItem:
    """Uploads an overlay layer into a ring renderer."""

    renderer: "RingBufferRenderer"
    overlay_layer: ScanlineOverlayLayer

    def prepare(self) -> None:
        """No CPU-bound prep yet; uploads run synchronously."""

    def upload(self) -> None:
        self.renderer.update_from_overlay_layer(self.overlay_layer)


@dataclass
class TextureUploadItem:
    """Uploads the artifact's texture layers into the texture renderer."""

    renderer: Optional["TextureArtifactRenderer"]
    artifact: CompiledSceneArtifact

    def prepare(self) -> None:
        """No CPU-bound prep yet; uploads run synchronously."""

    def upload(self) -> None:
        if self.renderer is not None:
            self.renderer.update_from_artifact(self.artifact)


class SceneRenderer(BaseRenderer):
    """Owns the GPU renderers, shaders and collections for the 3D scene."""

    def __init__(self):
        super().__init__()
        self.main_shader: Shader | None = None
        self.text_shader: Shader | None = None
        self.texture_shader: Shader | None = None
        self.background_shader: Shader | None = None
        self.shader_set: ShaderSet | None = None

        self.axis_renderer: AxisRenderer3D | None = None
        self.background_renderer: BackgroundRenderer | None = (
            BackgroundRenderer()
        )
        self.texture_renderer: TextureArtifactRenderer | None = None
        self.zone_renderer: ZoneRenderer | None = None
        self.laser_beam_renderer: LaserBeamRenderer | None = (
            LaserBeamRenderer()
        )

        self.ops_renderers: list[OpsRenderer] = []
        self.ring_renderers: list[RingBufferRenderer] = []
        self.cylinder_renderers: dict[float, CylinderRenderer] = {}
        self.model_renderers: list[ModelRenderer] = []
        self.had_rotary_layers = False
        self.cylinder_transform = np.eye(4, dtype=np.float64)

        self._viewport: ViewportConfig | None = None
        self._font_family: str | None = None

        # Ordered list of (renderer, shader_keys) in draw order.  The
        # deferred ring passes come after the texture renderer so rings
        # draw on top of the textures during playback.
        self.render_registry: list[tuple[BaseRenderer, tuple[str, ...]]] = []

    def _rebuild_registry(self) -> None:
        """Rebuilds the render registry from the current children."""
        registry: list[tuple[BaseRenderer, tuple[str, ...]]] = []

        # Draw the world first.
        if self.background_renderer is not None:
            registry.append((self.background_renderer, ("background",)))
        if self.axis_renderer is not None:
            registry.append((self.axis_renderer, ("main", "text")))

        # Draw the hardware.
        if self.zone_renderer is not None:
            registry.append((self.zone_renderer, ("main",)))
        for renderer in self.model_renderers:
            registry.append((renderer, ("main",)))
        for renderer in self.cylinder_renderers.values():
            registry.append((renderer, ("main",)))

        # Draw the ops and textures.
        if self.texture_renderer is not None:
            registry.append((self.texture_renderer, ("texture",)))
        for renderer in self.ops_renderers:
            registry.append((renderer, ("main",)))
        for renderer in self.ring_renderers:
            registry.append((renderer, ("main",)))
        if self.laser_beam_renderer is not None:
            registry.append((self.laser_beam_renderer, ("main",)))
        self.render_registry = registry

    def set_cylinder_transform(self, transform: np.ndarray):
        """Stores the assembly's cylinder base transform."""
        self.cylinder_transform = transform

    def set_viewport(self, viewport: ViewportConfig):
        """Stores the viewport config used to build children in init_gl."""
        self._viewport = viewport

    def set_font_family(self, font_family: str):
        """Stores the font family used to build the axis labels."""
        self._font_family = font_family

    def init_gl(self):
        """Creates and initializes all scene shaders and renderers."""
        viewport = self._viewport
        if viewport is None:
            viewport = ViewportConfig.default()
        font_family = self._font_family or "sans-serif"
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
        except GLError as e:
            logger.warning(
                "Background renderer init failed, "
                "falling back to clear color: %s",
                e,
            )
            self.background_renderer = None
        self.zone_renderer = ZoneRenderer()
        self.zone_renderer.init_gl()

        for renderer in (
            self.axis_renderer,
            self.background_renderer,
            self.texture_renderer,
            self.zone_renderer,
            self.laser_beam_renderer,
        ):
            if renderer is not None:
                self._add_child_renderer(renderer)

        self._rebuild_registry()

    def _cleanup_self(self):
        """Cleans up dynamically-rebuilt collections and shaders.

        Static children (axis, background, texture, zone, laser) are
        cleaned automatically by the base ``cleanup()`` walking
        ``_owned_renderers``.  Only the rebuilt collections and the
        owned shaders need manual cleanup here.
        """
        for renderer in self.ops_renderers:
            renderer.cleanup()
        for renderer in self.ring_renderers:
            renderer.cleanup()
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
        self._remove_child_renderer(self.axis_renderer)
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
        self._add_child_renderer(self.axis_renderer)
        self._rebuild_registry()
        return True

    def update_cylinders_from_doc(self, doc, viewport, machine):
        """Reads chuck diameters from the assembly and rebuilds cylinders."""
        desired_diameters: dict[float, bool] = {}
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

    def update_color_luts(self, provider: ColorLutProvider | None):
        """Fans out the shared colour LUT provider to all consumers."""
        if provider is None:
            return
        for renderer in self.ops_renderers:
            renderer.update_color_lut_from(provider)
        for renderer in self.ring_renderers:
            renderer.update_color_lut_from(provider)

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
        """Rebuilds the per-layer ops/ring renderers from an artifact."""
        for renderer in self.ops_renderers:
            renderer.cleanup()
        for renderer in self.ring_renderers:
            renderer.cleanup()
        self.ops_renderers.clear()
        self.ring_renderers.clear()

        for vl in artifact.vertex_layers:
            ops = OpsRenderer(is_rotary=vl.is_rotary)
            ops.init_gl()
            ops.update_from_vertex_layer(vl, show_travel_moves)
            ops.powered_offsets = vl.powered_cmd_offsets
            ops.travel_offsets = vl.travel_cmd_offsets
            self.ops_renderers.append(ops)

            ring = RingBufferRenderer(is_rotary=vl.is_rotary)
            ring.init_gl()
            ol = match_overlay_layer(artifact.overlay_layers, vl.is_rotary)
            if ol is not None:
                ring.update_from_overlay_layer(ol)
                ring.ring_offsets = ol.cmd_offsets
            else:
                ring.clear()
                ring.ring_offsets = np.array([], dtype=np.int32)
            self.ring_renderers.append(ring)

        if self.texture_renderer:
            self.texture_renderer.update_from_artifact(artifact)
        self._rebuild_registry()

    def prepare_chunked_upload(
        self, artifact: CompiledSceneArtifact, show_travel_moves: bool
    ) -> list[UploadItem]:
        """Creates fresh per-layer renderers and returns upload items."""
        for renderer in self.ops_renderers:
            renderer.cleanup()
        for renderer in self.ring_renderers:
            renderer.cleanup()
        self.ops_renderers.clear()
        self.ring_renderers.clear()

        upload_items: list[UploadItem] = []

        for vl in artifact.vertex_layers:
            ops = OpsRenderer(is_rotary=vl.is_rotary)
            ops.init_gl()
            self.ops_renderers.append(ops)
            upload_items.append(OpsLayerUploadItem(ops, vl, show_travel_moves))

            ring = RingBufferRenderer(is_rotary=vl.is_rotary)
            ring.init_gl()
            self.ring_renderers.append(ring)

        for ol in artifact.overlay_layers:
            for ring in self.ring_renderers:
                if ring.is_rotary == ol.is_rotary:
                    upload_items.append(OverlayLayerUploadItem(ring, ol))
                    break

        upload_items.append(TextureUploadItem(self.texture_renderer, artifact))
        self._rebuild_registry()
        return upload_items

    def upload_chunk(self, item: UploadItem) -> None:
        """Processes one prepared per-layer upload item."""
        item.upload()

    def clear_layers(self) -> None:
        """Clears all per-layer ops/ring/texture GPU buffers."""
        for renderer in self.ops_renderers:
            renderer.clear()
        for renderer in self.ring_renderers:
            renderer.clear()
        if self.texture_renderer:
            self.texture_renderer.clear()

    def extract_playback_offsets(self, artifact: CompiledSceneArtifact):
        """Stores each renderer's playback offsets from an artifact."""
        for renderer in self.ops_renderers:
            vl = match_vertex_layer(artifact.vertex_layers, renderer.is_rotary)
            if vl is not None:
                renderer.powered_offsets = vl.powered_cmd_offsets
                renderer.travel_offsets = vl.travel_cmd_offsets
            else:
                renderer.powered_offsets = np.array([], dtype=np.int32)
                renderer.travel_offsets = np.array([], dtype=np.int32)

        for renderer in self.ring_renderers:
            ol = match_overlay_layer(
                artifact.overlay_layers, renderer.is_rotary
            )
            if ol is not None:
                renderer.ring_offsets = ol.cmd_offsets
            else:
                renderer.ring_offsets = np.array([], dtype=np.int32)

    def prepare(self, ctx: RenderContext) -> None:
        """
        Prepares every registry renderer for the current frame.

        Runs the ``prepare`` phase of each renderer in the registry so
        that frame-level cross-dependencies (e.g. the laser point light
        feeding the model renderers) resolve before any draw.
        """
        for renderer, _ in self.render_registry:
            renderer.prepare(ctx)

    def render(
        self,
        ctx: RenderContext,
        shaders: ShaderSet | None = None,
        **kwargs,
    ) -> None:
        """
        Renders the whole scene for one frame via the render registry.

        All per-frame state is read from ``ctx`` (populated by the
        caller).  The root composite owns the shaders, so it uses
        ``self.shader_set`` and passes them to each registry renderer
        under ``render_pass`` state isolation.
        """
        for shader in (
            self.main_shader,
            self.text_shader,
            self.texture_shader,
            self.background_shader,
        ):
            if shader:
                shader.reset_uniforms()

        shaders = self.shader_set if shaders is None else shaders
        if shaders is None:
            return

        for renderer, shader_keys in self.render_registry:
            pass_shaders = tuple(getattr(shaders, key) for key in shader_keys)
            with render_pass(*pass_shaders):
                renderer.render(ctx, shaders)
