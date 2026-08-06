"""
A composite renderer owning all scene GPU resources for the 3D canvas.

The SceneRenderer owns the child renderers and shaders plus the per-layer
collections (layer groups, cylinders, models).  Canvas3D is responsible for
the GL context lifecycle and per-frame state; it delegates resource creation,
rebuilds and theme/colour application to this class.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....machine.assembly import LinkRole
from ....machine.models.laser import LaserHead
from ....shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from ....simulator.machine_state import MachineState
from ....simulator.scene3d import CompiledSceneArtifact
from ..color_lut_provider import ColorLutProvider
from ..gl_utils import rotation_4x4
from ..layer_renderer_group import (
    LayerRendererGroup,
    match_overlay_layer,
    match_vertex_layer,
)
from ..shader import Shader, SimpleShader, TextShader, TextureShader
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


class SceneRenderer(BaseRenderer):
    """Owns the GPU renderers, shaders and collections for the 3D scene."""

    def __init__(self):
        super().__init__()
        self.main_shader: Optional[Shader] = None
        self.text_shader: Optional[Shader] = None
        self.texture_shader: Optional[Shader] = None

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
        self.model_renderers: List[Tuple[ModelRenderer, str]] = []
        self.had_rotary_layers = False
        self.cylinder_transform = np.eye(4, dtype=np.float64)

    def set_cylinder_transform(self, transform: np.ndarray):
        """Stores the assembly's cylinder base transform."""
        self.cylinder_transform = transform

    def init_gl(self, viewport: ViewportConfig, font_family: str):
        """Creates and initializes all scene shaders and renderers."""
        self.main_shader = SimpleShader()
        self.text_shader = TextShader()
        self.texture_shader = TextureShader()

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
        for renderer, _ in self.model_renderers:
            renderer.cleanup()
        if self.main_shader:
            self.main_shader.cleanup()
        if self.text_shader:
            self.text_shader.cleanup()
        if self.texture_shader:
            self.texture_shader.cleanup()

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
        for r, _ in self.model_renderers:
            r.cleanup()
        self.model_renderers.clear()

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

            renderer = ModelRenderer(resolved)
            renderer.init_gl()
            logger.debug(
                "Model renderer created: vao=%d, vertex_count=%d, bounds=%s",
                renderer._vao,
                renderer._vertex_count,
                renderer.bounds,
            )
            self.model_renderers.append((renderer, link.name))

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

    def render(
        self,
        ctx,
        op_player,
        viewport: ViewportConfig,
        machine,
        compiled_artifact,
        doc,
        show_grid: bool = True,
        show_nogo_zones: bool = True,
        show_models: bool = True,
    ):
        """Renders the whole scene for one frame."""
        if self.background_renderer:
            self.background_renderer.render(ctx)

        mvp_ui = ctx.mvp_ui
        margin_shift = ctx.margin_shift
        mvp_ui_gl = ctx.mvp_ui.T

        if (
            self.axis_renderer is not None
            and self.main_shader is not None
            and self.text_shader is not None
            and show_grid
        ):
            self.axis_renderer.render(
                ctx,
                self.main_shader,
                self.text_shader,
                ctx.mvp_scene.T,
                mvp_ui_gl,
                origin_offset_mm=viewport.wcs_offset_mm,
                x_right=viewport.x_right,
                y_down=viewport.y_down,
                x_negative=viewport.x_negative,
                y_negative=viewport.y_negative,
            )

        if self.zone_renderer and self.main_shader and show_nogo_zones:
            zone_mvp_gl = (mvp_ui @ margin_shift).T
            self.zone_renderer.render(ctx, self.main_shader, zone_mvp_gl)

        # Compute cylinder rotation from mapped ops.
        #
        # Degrees are stored in extra_axes by KinematicMapping and
        # copied into state.axes by apply_command.  The rotary_axis
        # property tells us which axis holds the angle.
        cyl_angle = 0.0
        ra = None
        if op_player:
            ra = op_player.rotary_axis
        vis_rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if op_player and self.had_rotary_layers and machine and ra is not None:
            asm = machine.assembly
            if asm.has_rotary:
                degrees = op_player.state.axes.get(ra, 0.0)
                cyl_angle = math.radians(degrees)

        if self.had_rotary_layers and machine:
            cyl_base_mvp = (
                mvp_ui.astype(np.float64)
                @ margin_shift.astype(np.float64)
                @ self.cylinder_transform
            )
        else:
            cyl_base_mvp = mvp_ui.astype(np.float64) @ margin_shift.astype(
                np.float64
            )

        rot_4x4 = rotation_4x4(vis_rot_axis, cyl_angle)
        rot_cyl_gl = (cyl_base_mvp @ rot_4x4).T.astype(np.float32)

        tex_reached = None
        if op_player and compiled_artifact:
            tex_reached = 0
            playhead = op_player.current_index
            for tl in compiled_artifact.texture_layers:
                if playhead >= tl.activation_cmd_idx:
                    tex_reached += 1

        # Laser beams: computed now so models can use the point light.
        # The beam itself is drawn after the scanline overlay but before
        # the models, so the laser head model stays in front.
        if (
            op_player
            and self.main_shader
            and machine
            and self.laser_beam_renderer
        ):
            self.laser_beam_renderer.update_from_state(
                op_player.state,
                machine,
                viewport,
                margin_shift,
                ra,
                doc,
                op_player,
            )
            laser_light_pos = self.laser_beam_renderer.laser_light_pos
        else:
            laser_light_pos = None

        deferred_ring_renders = []
        for group in self.layer_groups:
            if self.main_shader:
                ring = group.render(
                    ctx,
                    self.main_shader,
                    op_player,
                    mvp_ui_gl,
                    rot_cyl_gl,
                )
                if ring is not None:
                    deferred_ring_renders.append(ring)

        if self.cylinder_renderers and self.main_shader:
            cyl_mesh_mvp = (
                mvp_ui @ margin_shift @ self.cylinder_transform @ rot_4x4
            ).astype(np.float64)
            cyl_mvp_gl = cyl_mesh_mvp.T.astype(np.float32)
            for renderer in self.cylinder_renderers.values():
                renderer.update_from_state(cyl_mvp_gl)
                renderer.render(ctx, self.main_shader)

        if self.texture_renderer and self.texture_shader:
            rot_cyl_mvp = cyl_base_mvp @ rot_4x4
            self.texture_renderer.update_from_state(mvp_ui, rot_cyl_mvp)
            self.texture_renderer.render(
                ctx, self.texture_shader, reached_count=tex_reached
            )
            self.texture_renderer.render_cylinder(
                ctx, self.texture_shader, reached_count=tex_reached
            )

        for ring_renderer, mvp, exec_ring in deferred_ring_renders:
            if self.main_shader:
                ring_renderer.render(
                    ctx,
                    self.main_shader,
                    mvp,
                    executed_vertex_count=exec_ring,
                )

        if self.laser_beam_renderer and self.main_shader:
            self.laser_beam_renderer.render(ctx, self.main_shader)

        if show_models and self.model_renderers and machine:
            asm = machine.assembly
            wcs = viewport.wcs_offset_mm
            model_state = op_player.state if op_player else MachineState()
            model_transforms = asm.model_world_transforms(
                model_state, wcs_offset=wcs
            )
            is_rotary = ra is not None and asm.has_rotary
            layer_rotary_diameter = 0.0
            if is_rotary and op_player:
                current_layer = op_player.get_current_layer(doc)
                if current_layer:
                    layer_rotary_diameter = current_layer.rotary_diameter
            for renderer, link_name in self.model_renderers:
                if self.main_shader:
                    t = model_transforms.get(link_name)
                    if t is None:
                        continue
                    module_transform = t.astype(np.float32)
                    if is_rotary:
                        link = asm.get_link(link_name)
                        if link and link.role != LinkRole.CHUCK:
                            focal = 50.0
                            if link.name.startswith("head_"):
                                try:
                                    idx = int(link.name.split("_")[1])
                                    laser = machine.heads[idx]
                                    if isinstance(laser, LaserHead):
                                        if laser.focal_distance > 0:
                                            focal = laser.focal_distance
                                except (ValueError, IndexError):
                                    pass
                            rotary_heads = asm.head_rotary_positions(
                                model_state,
                                layer_rotary_diameter,
                                focal_distance=focal,
                            )
                            if link.name in rotary_heads:
                                pos = rotary_heads[link.name]
                                module_transform[:3, 3] = pos.astype(
                                    np.float32
                                )
                    combined = mvp_ui @ margin_shift @ module_transform
                    renderer.update_from_state(
                        combined.T,
                        model_matrix=margin_shift @ module_transform,
                        point_light_pos=laser_light_pos,
                    )
                    renderer.render(ctx, self.main_shader)
