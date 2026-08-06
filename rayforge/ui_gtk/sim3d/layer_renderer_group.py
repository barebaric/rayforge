"""A group of GPU renderers owned for one compiled scene layer."""

import logging
from typing import Any, List, Optional

from ...simulator.scene3d import ScanlineOverlayLayer, VertexLayer
from .renderer.ops_renderer import OpsRenderer
from .renderer.ring_buffer_renderer import RingBufferRenderer

logger = logging.getLogger(__name__)


class LayerRendererGroup:
    """Owns the GPU renderers and playback offsets for one compiled layer."""

    def __init__(self, is_rotary: bool):
        self.is_rotary = is_rotary
        self.ops_renderer = OpsRenderer()
        self.ring_renderer = RingBufferRenderer()
        self.powered_offsets: Any = []
        self.travel_offsets: Any = []
        self.ring_offsets: Any = []

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

    def render(
        self,
        ctx,
        shader,
        op_player,
        mvp_flat_gl,
        mvp_rot_gl,
    ) -> Optional[tuple]:
        """
        Renders the group's ops and returns a deferred ring draw or None.

        The ring buffer is drawn after the textures during playback, so it
        is returned for the scene renderer to render later.
        """
        mvp = mvp_rot_gl if self.is_rotary else mvp_flat_gl

        exec_powered = -1
        exec_travel = -1
        exec_ring = -1
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

        if shader:
            self.ops_renderer.render(
                ctx,
                shader,
                mvp,
                executed_vertex_count=exec_powered,
                executed_travel_vertex_count=exec_travel,
            )

        if self.ring_renderer.vertex_count > 0 and shader:
            tag = "rot" if self.is_rotary else "flat"
            logger.debug(
                f"[RING-PLAYBACK] {tag} "
                f"exec={exec_ring} "
                f"total={self.ring_renderer.vertex_count}"
            )
            return (self.ring_renderer, mvp, exec_ring)
        return None

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
