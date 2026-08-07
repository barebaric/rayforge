"""A group of GPU renderers owned for one compiled scene layer."""

import logging
from typing import Any, List, Optional

from ...simulator.scene3d import ScanlineOverlayLayer, VertexLayer
from .gl_utils import RenderContext, ShaderSet
from .renderer.ops_renderer import OpsRenderer
from .renderer.ring_buffer_renderer import RingBufferRenderer

logger = logging.getLogger(__name__)


class LayerRendererGroup:
    """Owns the GPU renderers and playback offsets for one compiled layer."""

    def __init__(self, is_rotary: bool):
        self.is_rotary = is_rotary
        self.ops_renderer = OpsRenderer(is_rotary=is_rotary)
        self.ring_renderer = RingBufferRenderer(is_rotary=is_rotary)
        self.powered_offsets: Any = []
        self.travel_offsets: Any = []
        self.ring_offsets: Any = []
        self._exec_powered = -1
        self._exec_travel = -1
        self._exec_ring = -1

    def init_gl(self):
        self.ops_renderer.init_gl()
        self.ring_renderer.init_gl()

    def update_from_artifact(
        self,
        vl: VertexLayer,
        ol: Optional[ScanlineOverlayLayer],
        show_travel_moves: bool,
    ):
        """Ingests a compiled vertex layer and its overlay into the group."""
        self.ops_renderer.update_from_vertex_layer(vl, show_travel_moves)
        self.powered_offsets = vl.powered_cmd_offsets
        self.travel_offsets = vl.travel_cmd_offsets

        if ol is not None:
            self.ring_renderer.update_from_overlay_layer(ol)
            self.ring_offsets = ol.cmd_offsets
        else:
            self.ring_renderer.clear()
            self.ring_offsets = []

    def cleanup(self):
        self.ops_renderer.cleanup()
        self.ring_renderer.cleanup()

    def prepare(self, ctx: RenderContext) -> None:
        """
        Computes the executed-vertex counts for this frame.

        Reads the playhead from ``ctx.op_player`` and maps it through the
        group's command offsets, stashing the resulting counts so the ops
        and ring draws can consume them.
        """
        exec_powered = -1
        exec_travel = -1
        exec_ring = -1
        op_player = ctx.op_player
        if op_player:
            idx = op_player.current_index
            if len(self.powered_offsets) > 0 and idx + 1 < len(
                self.powered_offsets
            ):
                exec_powered = self.powered_offsets[idx + 1]
            if len(self.travel_offsets) > 0 and idx + 1 < len(
                self.travel_offsets
            ):
                exec_travel = self.travel_offsets[idx + 1]
            if len(self.ring_offsets) > 0 and idx + 1 < len(self.ring_offsets):
                exec_ring = self.ring_offsets[idx + 1]

            pv_total = self.ops_renderer.powered_vertex_count
            off_len = len(self.powered_offsets)
            if exec_powered >= 0 and idx % 50 == 0:
                tag = "rot" if self.is_rotary else "flat"
                logger.info(
                    f"[PLAYBACK-DIAG] {tag} "
                    f"idx={idx}/{off_len - 1} "
                    f"exec={exec_powered}/{pv_total} "
                    f"off[-3:]="
                    f"{self.powered_offsets[-3:]}"
                )

        self._exec_powered = exec_powered
        self._exec_travel = exec_travel
        self._exec_ring = exec_ring

    def render(self, ctx: RenderContext, shaders: ShaderSet) -> None:
        """Renders the group's ops (toolpaths)."""
        ctx.executed_vertex_count = self._exec_powered
        ctx.executed_travel_vertex_count = self._exec_travel
        self.ops_renderer.render(ctx, shaders)

    def render_ring(self, ctx: RenderContext, shaders: ShaderSet) -> None:
        """Renders the group's ring buffer (after the textures)."""
        if self.ring_renderer.vertex_count <= 0:
            return
        tag = "rot" if self.is_rotary else "flat"
        logger.debug(
            f"[RING-PLAYBACK] {tag} "
            f"exec={self._exec_ring} "
            f"total={self.ring_renderer.vertex_count}"
        )
        ctx.executed_vertex_count = self._exec_ring
        self.ring_renderer.render(ctx, shaders)

    def clear(self):
        self.ops_renderer.clear()
        self.ring_renderer.clear()
        self.powered_offsets = []
        self.travel_offsets = []
        self.ring_offsets = []


def match_vertex_layer(
    vertex_layers: List[VertexLayer], is_rotary: bool
) -> Optional[VertexLayer]:
    """Returns the vertex layer matching the given rotary flag."""
    for vl in vertex_layers:
        if vl.is_rotary == is_rotary:
            return vl
    return None


def match_overlay_layer(
    overlay_layers: List[ScanlineOverlayLayer], is_rotary: bool
) -> Optional[ScanlineOverlayLayer]:
    """Returns the overlay layer matching the given rotary flag."""
    for ol in overlay_layers:
        if ol.is_rotary == is_rotary:
            return ol
    return None
