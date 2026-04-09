from __future__ import annotations

import logging
import math
from copy import deepcopy
from enum import auto, Enum
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

from gettext import gettext as _

from rayforge.core.geo.bezier import linearize_bezier
from rayforge.core.geo.constants import (
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    CMD_TYPE_LINE,
)
from rayforge.core.geo.types import Point3D
from rayforge.core.ops import (
    BezierToCommand,
    Command,
    CurveToCommand,
    LineToCommand,
    MoveToCommand,
    MovingCommand,
    Ops,
    OpsSection,
    SectionType,
    SetPowerCommand,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import OpsTransformer, ExecutionPhase
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.geo import Geometry

logger = logging.getLogger(__name__)


class _EventType(Enum):
    ENTER_TAB = auto()
    EXIT_TAB = auto()


# Type aliases for clarity instead of raw tuples.


class _ClipPoint(NamedTuple):
    x: float
    y: float
    width: float


class _SubpathKey(NamedTuple):
    section_idx: int
    subpath_idx: int


class _TabRegion(NamedTuple):
    start: float
    end: float


class TabOpsTransformer(OpsTransformer):
    """
    Creates gaps in toolpaths by finding the closest point on the path for
    each tab and creating a precise cut. This is robust against prior ops
    transformations and avoids clipping unrelated paths that may be nearby.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    @property
    def execution_phase(self) -> ExecutionPhase:
        """Tabs intentionally break paths, so they must run after smoothing."""
        return ExecutionPhase.PATH_INTERRUPTION

    @property
    def label(self) -> str:
        return _("Tabs")

    @property
    def description(self) -> str:
        return _(
            "Creates holding tabs by adding gaps or reducing power "
            "on cut paths"
        )

    def _generate_tab_clip_data(
        self, workpiece: WorkPiece
    ) -> List[_ClipPoint]:
        """
        Generates clip data (center point and width) for each tab in the
        workpiece's local coordinate space. This matches the coordinate space
        of the incoming Ops object during the generation phase.
        """
        if not workpiece.boundaries or workpiece.boundaries.is_empty():
            logger.debug(
                "TabOps: workpiece has no vectors, cannot generate clip data."
            )
            return []

        clip_data: List[_ClipPoint] = []

        logger.debug(
            "TabOps: Generating clip data in LOCAL space for workpiece "
            f"'{workpiece.name}'"
        )
        logger.debug(
            f"TabOps: Workpiece vectors bbox: {workpiece.boundaries.rect()}"
        )

        for tab in workpiece.tabs:
            cmd = workpiece.boundaries.get_command_at(tab.segment_index)
            if cmd is None:
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            cmd_type, x, y, z, p1, p2, p3, p4 = cmd
            end_point = (x, y, z)

            if cmd_type not in (CMD_TYPE_LINE, CMD_TYPE_ARC, CMD_TYPE_BEZIER):
                continue

            p_start_3d: Point3D = (0.0, 0.0, 0.0)
            if tab.segment_index > 0:
                prev_cmd = workpiece.boundaries.get_command_at(
                    tab.segment_index - 1
                )
                if prev_cmd:
                    _, prev_x, prev_y, prev_z, _, _, _, _ = prev_cmd
                    p_start_3d = (prev_x, prev_y, prev_z)

            logger.debug(
                f"Processing Tab UID {tab.uid} on segment "
                f"{tab.segment_index} "
                f"(type: {cmd_type}) starting from {p_start_3d}"
            )

            center_x, center_y = 0.0, 0.0

            if cmd_type == CMD_TYPE_LINE:
                p_start, p_end = p_start_3d[:2], end_point[:2]
                center_x = p_start[0] + (p_end[0] - p_start[0]) * tab.pos
                center_y = p_start[1] + (p_end[1] - p_start[1]) * tab.pos

            elif cmd_type == CMD_TYPE_ARC:
                i, j, cw = p1, p2, p3
                center_offset = (i, j)
                clockwise = bool(cw)
                center = (
                    p_start_3d[0] + center_offset[0],
                    p_start_3d[1] + center_offset[1],
                )
                radius = math.dist(p_start_3d[:2], center)
                if radius < 1e-9:
                    continue

                start_angle = math.atan2(
                    p_start_3d[1] - center[1],
                    p_start_3d[0] - center[0],
                )
                end_angle = math.atan2(
                    end_point[1] - center[1],
                    end_point[0] - center[0],
                )
                angle_range = end_angle - start_angle
                if clockwise:
                    if angle_range > 0:
                        angle_range -= 2 * math.pi
                else:
                    if angle_range < 0:
                        angle_range += 2 * math.pi

                tab_angle = start_angle + angle_range * tab.pos
                center_x = center[0] + radius * math.cos(tab_angle)
                center_y = center[1] + radius * math.sin(tab_angle)

            elif cmd_type == CMD_TYPE_BEZIER:
                c1x, c1y, c2x, c2y = p1, p2, p3, p4
                t = tab.pos
                t2 = t * t
                t3 = t2 * t
                mt = 1.0 - t
                mt2 = mt * mt
                mt3 = mt2 * mt
                center_x = (
                    mt3 * p_start_3d[0]
                    + 3.0 * mt2 * t * c1x
                    + 3.0 * mt * t2 * c2x
                    + t3 * end_point[0]
                )
                center_y = (
                    mt3 * p_start_3d[1]
                    + 3.0 * mt2 * t * c1y
                    + 3.0 * mt * t2 * c2y
                    + t3 * end_point[1]
                )

            logger.debug(
                f"Local space tab center (from normalized vectors): "
                f"({center_x:.4f}, {center_y:.4f}), "
                f"width: {tab.width:.2f}mm"
            )
            clip_data.append(_ClipPoint(center_x, center_y, tab.width))

        logger.debug(f"TabOps: Finished generating clip data: {clip_data}")
        return clip_data

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict] = None,
    ) -> None:
        if not self.enabled:
            return
        if not workpiece:
            logger.debug("TabOpsTransformer: No workpiece provided, skipping.")
            return
        if not workpiece.tabs_enabled or not workpiece.tabs:
            logger.debug(
                "TabOpsTransformer: Tabs disabled or no tabs on workpiece "
                f"'{workpiece.name}', skipping."
            )
            return

        tab_power = 0.0
        if settings:
            tab_power = settings.get("tab_power", 0.0)

        logger.debug(
            f"TabOpsTransformer running for workpiece '{workpiece.name}' "
            f"with {len(workpiece.tabs)} tabs, tab_power={tab_power}."
        )

        tab_clip_data = self._generate_tab_clip_data(workpiece)
        if not tab_clip_data:
            logger.debug("No tab clip data was generated. Skipping clipping.")
            return

        logger.debug(
            f"Generated {len(tab_clip_data)} tab clip points for clipping."
        )

        processed_clip_data = tab_clip_data
        # Check if the coordinate space of the workpiece vectors (where tabs
        # are defined) is different from the final workpiece size (which the
        # incoming Ops object should represent). If so, we must scale the tab
        # points.
        # This handles cases like FrameProducer where Ops are generated at
        # final size, but the workpiece.boundaries are still normalized to a
        # 1x1 box.
        if workpiece.boundaries and not workpiece.boundaries.is_empty():
            vector_rect = workpiece.boundaries.rect()
            if vector_rect:
                final_w, final_h = workpiece.size
                _vx, _vy, vector_w, vector_h = vector_rect

                # Avoid division by zero for empty or linear geometry
                if vector_w > 1e-6 and vector_h > 1e-6:
                    scale_x = final_w / vector_w
                    scale_y = final_h / vector_h

                    # Only apply scaling if it's significant, to avoid
                    # float errors
                    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                        logger.debug(
                            "TabOps: Scaling tab clip points from vector "
                            "space to final ops space. "
                            f"Scale=({scale_x:.3f}, {scale_y:.3f})"
                        )
                        processed_clip_data = [
                            _ClipPoint(
                                cp.x * scale_x,
                                cp.y * scale_y,
                                cp.width,
                            )
                            for cp in tab_clip_data
                        ]

        logger.debug(
            f"TabOps: Clipping points to be used: {processed_clip_data}"
        )

        if tab_power > 0:
            self._apply_tab_power(
                ops, processed_clip_data, tab_power, settings
            )
        else:
            self._apply_tab_gaps(ops, processed_clip_data)

    @staticmethod
    def _split_into_subpaths(
        cmds: List[Command],
    ) -> List[List[Command]]:
        """Split commands into subpaths at MoveToCommand boundaries."""
        temp_ops = Ops()
        temp_ops.replace_all(cmds)
        return temp_ops.split_into_subpaths()

    def _assign_clips_globally(
        self,
        sections: List[OpsSection],
        clip_data: List[_ClipPoint],
    ) -> Dict[_SubpathKey, List[_ClipPoint]]:
        """
        Assign each clip to exactly ONE subpath across ALL sections,
        choosing the globally closest subpath. This prevents a single
        tab from clipping multiple contours (e.g. inner and outer
        contours produced by ContourProducer).
        """
        # Collect all subpaths from VECTOR_OUTLINE sections.
        all_subpaths: List[Tuple[_SubpathKey, List[Command]]] = []
        for sec_idx, section in enumerate(sections):
            if section.section_type != SectionType.VECTOR_OUTLINE:
                continue
            for sp_idx, sp_cmds in enumerate(
                self._split_into_subpaths(section.commands)
            ):
                all_subpaths.append((_SubpathKey(sec_idx, sp_idx), sp_cmds))

        assignments: Dict[_SubpathKey, List[_ClipPoint]] = {}

        # For each clip, find the single closest subpath globally.
        for clip in clip_data:
            best_key: Optional[_SubpathKey] = None
            best_dist = float("inf")

            for key, sp_cmds in all_subpaths:
                sp_ops = Ops()
                # Create a distinct copy for each evaluation to avoid
                # shared references.
                sp_ops.replace_all(deepcopy(sp_cmds))
                sp_ops.preload_state()
                geo = sp_ops.to_geometry()
                closest = geo.find_closest_point(clip.x, clip.y)
                if closest:
                    _, _, pt = closest
                    d = (clip.x - pt[0]) ** 2 + (clip.y - pt[1]) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_key = key

            # Only assign if the closest subpath is within a reasonable
            # distance of the clip point.
            if best_key is not None and best_dist <= (clip.width * 2) ** 2:
                assignments.setdefault(best_key, []).append(clip)

        return assignments

    def _apply_tab_gaps(
        self,
        ops: Ops,
        clip_data: List[_ClipPoint],
    ) -> None:
        sections = list(ops.iter_sections())
        assignments = self._assign_clips_globally(sections, clip_data)

        new_commands: List[Command] = []
        for sec_idx, section in enumerate(sections):
            # Preserve the section markers (start/end commands).
            new_commands.extend(section.markers)

            if section.section_type != SectionType.VECTOR_OUTLINE:
                # For any other section type, or commands outside a
                # section, pass them through unmodified.
                new_commands.extend(section.commands)
                continue

            subpaths = self._split_into_subpaths(section.commands)
            for sp_idx, sp_cmds in enumerate(subpaths):
                key = _SubpathKey(sec_idx, sp_idx)
                clips = assignments.get(key, [])
                if clips:
                    has_curves = any(
                        isinstance(c, CurveToCommand) for c in sp_cmds
                    )
                    if has_curves:
                        processed = self._clip_subpath_with_gaps(
                            sp_cmds, clips
                        )
                        new_commands.extend(processed)
                    else:
                        sp_ops = Ops()
                        sp_ops.replace_all(deepcopy(sp_cmds))
                        sp_ops.preload_state()
                        for clip in clips:
                            sp_ops.clip_at(clip.x, clip.y, clip.width)
                        new_commands.extend(sp_ops.commands)
                else:
                    new_commands.extend(sp_cmds)

        ops.replace_all(new_commands)

    def _apply_tab_power(
        self,
        ops: Ops,
        clip_data: List[_ClipPoint],
        tab_power: float,
        settings: Optional[Dict],
    ) -> None:
        original_power = settings.get("power", 1.0) if settings else 1.0
        actual_tab_power = tab_power * original_power

        sections = list(ops.iter_sections())
        assignments = self._assign_clips_globally(sections, clip_data)

        new_commands: List[Command] = []
        for sec_idx, section in enumerate(sections):
            # Preserve the section markers (start/end commands).
            new_commands.extend(section.markers)

            if section.section_type != SectionType.VECTOR_OUTLINE:
                # For any other section type, or commands outside a
                # section, pass them through unmodified.
                new_commands.extend(section.commands)
                continue

            subpaths = self._split_into_subpaths(section.commands)
            for sp_idx, sp_cmds in enumerate(subpaths):
                key = _SubpathKey(sec_idx, sp_idx)
                clips = assignments.get(key, [])
                if clips:
                    processed = self._insert_power_commands(
                        sp_cmds, clips, actual_tab_power, original_power
                    )
                    new_commands.extend(processed)
                else:
                    new_commands.extend(sp_cmds)

        ops.replace_all(new_commands)

    def _insert_power_commands(
        self,
        commands: List[Command],
        clip_data: List[_ClipPoint],
        tab_power: float,
        original_power: float,
    ) -> List[Command]:
        has_curves = any(isinstance(c, CurveToCommand) for c in commands)
        if has_curves:
            return self._insert_power_commands_curve_aware(
                commands, clip_data, tab_power, original_power
            )

        temp_ops = Ops()
        temp_ops.replace_all(deepcopy(commands))
        temp_ops.preload_state()
        temp_ops.linearize_all()
        linear_cmds = temp_ops.commands

        if len(linear_cmds) < 2:
            return commands

        geo_cmds = [
            cmd
            for cmd in linear_cmds
            if isinstance(cmd, (MoveToCommand, LineToCommand))
        ]

        if len(geo_cmds) < 2:
            return commands

        tab_regions = self._compute_tab_regions(temp_ops, geo_cmds, clip_data)

        if not tab_regions:
            return commands

        tab_regions.sort(key=lambda r: r.start)

        return self._build_commands_with_power(
            linear_cmds, geo_cmds, tab_regions, tab_power, original_power
        )

    def _compute_tab_regions(
        self,
        temp_ops: Ops,
        geo_cmds: Sequence[Command],
        clip_data: List[_ClipPoint],
    ) -> List[_TabRegion]:
        tab_regions: List[_TabRegion] = []

        for clip in clip_data:
            temp_geo = temp_ops.to_geometry()
            closest = temp_geo.find_closest_point(clip.x, clip.y)
            if not closest:
                continue

            dist_sq = (clip.x - closest[2][0]) ** 2 + (
                clip.y - closest[2][1]
            ) ** 2
            if dist_sq > (clip.width * 2) ** 2:
                continue

            hit_dist = self._compute_hit_distance(geo_cmds, closest)
            if hit_dist is None:
                continue

            tab_regions.append(
                _TabRegion(
                    max(0.0, hit_dist - clip.width / 2.0),
                    hit_dist + clip.width / 2.0,
                )
            )

        return tab_regions

    def _compute_hit_distance(
        self,
        geo_cmds: Sequence[Command],
        closest: Tuple[int, float, Tuple[float, ...]],
    ) -> Optional[float]:
        segment_idx = closest[0]
        t = closest[1]

        if segment_idx >= len(geo_cmds):
            return None

        hit_dist = 0.0
        last_pos = geo_cmds[0].end
        if last_pos is None:
            return None

        for i in range(1, segment_idx):
            cmd = geo_cmds[i]
            if isinstance(cmd, MoveToCommand) and cmd.end:
                last_pos = cmd.end
            elif isinstance(cmd, LineToCommand) and cmd.end:
                hit_dist += math.dist(last_pos[:2], cmd.end[:2])
                last_pos = cmd.end

        hit_segment_cmd = geo_cmds[segment_idx]
        if isinstance(hit_segment_cmd, LineToCommand) and hit_segment_cmd.end:
            dist = math.dist(last_pos[:2], hit_segment_cmd.end[:2])
            hit_dist += t * dist
            return hit_dist

        return None

    def _build_commands_with_power(
        self,
        linear_cmds: List[Command],
        geo_cmds: Sequence[Command],
        tab_regions: List[_TabRegion],
        tab_power: float,
        original_power: float,
    ) -> List[Command]:
        result: List[Command] = []
        accum_dist = 0.0
        current_power = original_power
        last_pos = geo_cmds[0].end
        assert last_pos is not None

        result.append(deepcopy(linear_cmds[0]))

        for cmd in linear_cmds[1:]:
            if isinstance(cmd, LineToCommand):
                p1, p2 = last_pos, cmd.end
                seg_len = math.dist(p1[:2], p2[:2])

                if seg_len < 1e-9:
                    last_pos = p2
                    continue

                seg_start = accum_dist
                seg_end = accum_dist + seg_len

                events = self._collect_events(seg_start, seg_end, tab_regions)

                if events:
                    self._process_segment_events(
                        result,
                        cmd,
                        p1,
                        p2,
                        seg_len,
                        seg_start,
                        seg_end,
                        events,
                        tab_power,
                        original_power,
                        current_power,
                    )
                    current_power = self._get_final_power(
                        events, tab_power, original_power
                    )
                else:
                    result.append(deepcopy(cmd))

                last_pos = p2
                accum_dist += seg_len
            elif isinstance(cmd, MoveToCommand):
                result.append(deepcopy(cmd))
                if cmd.end:
                    last_pos = cmd.end
            else:
                result.append(deepcopy(cmd))

        return result

    def _collect_events(
        self,
        seg_start: float,
        seg_end: float,
        tab_regions: List[_TabRegion],
    ) -> List[Tuple[float, _EventType]]:
        events: List[Tuple[float, _EventType]] = []

        for region in tab_regions:
            if region.end <= seg_start or region.start >= seg_end:
                continue
            enter = max(region.start, seg_start)
            exit = min(region.end, seg_end)
            events.append((enter, _EventType.ENTER_TAB))
            events.append((exit, _EventType.EXIT_TAB))

        events.sort(key=lambda e: e[0])
        return events

    def _process_segment_events(
        self,
        result: List[Command],
        cmd: LineToCommand,
        p1: Point3D,
        p2: Point3D,
        seg_len: float,
        seg_start: float,
        seg_end: float,
        events: List[Tuple[float, _EventType]],
        tab_power: float,
        original_power: float,
        current_power: float,
    ) -> None:
        last_dist = seg_start

        for event_dist, event_type in events:
            if event_dist > last_dist + 1e-9:
                t = (event_dist - seg_start) / seg_len
                split_pt = self._interpolate_point(p1, p2, t)

                new_cmd = LineToCommand(split_pt)
                new_cmd.state = deepcopy(cmd.state) if cmd.state else None
                result.append(new_cmd)

            if event_type == _EventType.ENTER_TAB:
                if current_power != tab_power:
                    result.append(SetPowerCommand(tab_power))
                    current_power = tab_power
            else:
                if current_power != original_power:
                    result.append(SetPowerCommand(original_power))
                    current_power = original_power

            last_dist = event_dist

        if seg_end > last_dist + 1e-9:
            result.append(deepcopy(cmd))

    def _interpolate_point(
        self, p1: Point3D, p2: Point3D, t: float
    ) -> Point3D:
        return (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2])
            if len(p1) > 2 and len(p2) > 2
            else 0.0,
        )

    def _get_final_power(
        self,
        events: List[Tuple[float, _EventType]],
        tab_power: float,
        original_power: float,
    ) -> float:
        if not events:
            return original_power

        _, last_event_type = events[-1]
        if last_event_type == _EventType.EXIT_TAB:
            return original_power
        return tab_power

    # ------------------------------------------------------------------
    # Bezier-aware helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bezier_arc_length_2d(
        p0: Point3D,
        c1: Point3D,
        c2: Point3D,
        p1: Point3D,
    ) -> float:
        segments = linearize_bezier(p0, c1, c2, p1, 200)
        return sum(math.dist(s[:2], e[:2]) for s, e in segments)

    @staticmethod
    def _bezier_distance_to_t(
        p0: Point3D,
        c1: Point3D,
        c2: Point3D,
        p1: Point3D,
        target_dist: float,
    ) -> float:
        num_samples = 200
        segments = linearize_bezier(p0, c1, c2, p1, num_samples)
        accum = 0.0
        for i, (seg_start, seg_end) in enumerate(segments):
            seg_len = math.dist(seg_start[:2], seg_end[:2])
            if accum + seg_len >= target_dist - 1e-9:
                if seg_len < 1e-9:
                    return min(1.0, i / num_samples)
                local_t = max(0.0, min(1.0, (target_dist - accum) / seg_len))
                return min(1.0, (i + local_t) / num_samples)
            accum += seg_len
        return 1.0

    @staticmethod
    def _extract_bezier_subsegment_3d(
        p0: Point3D,
        c1: Point3D,
        c2: Point3D,
        p1: Point3D,
        t_start: float,
        t_end: float,
    ) -> Tuple[Point3D, Point3D, Point3D, Point3D]:
        def _lerp_3d(a: Point3D, b: Point3D, t: float) -> Point3D:
            return (
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            )

        def _subdivide_3d(
            a: Point3D,
            b: Point3D,
            c: Point3D,
            d: Point3D,
            t: float,
        ):
            m01 = _lerp_3d(a, b, t)
            m12 = _lerp_3d(b, c, t)
            m23 = _lerp_3d(c, d, t)
            m0112 = _lerp_3d(m01, m12, t)
            m1223 = _lerp_3d(m12, m23, t)
            sp = _lerp_3d(m0112, m1223, t)
            return (a, m01, m0112, sp), (sp, m1223, m23, d)

        if t_start <= 1e-9 and t_end >= 1.0 - 1e-9:
            return (p0, c1, c2, p1)
        if t_end >= 1.0 - 1e-9:
            _, right = _subdivide_3d(p0, c1, c2, p1, t_start)
            return right
        if t_start <= 1e-9:
            left, _ = _subdivide_3d(p0, c1, c2, p1, t_end)
            return left
        _, right = _subdivide_3d(p0, c1, c2, p1, t_start)
        s = (t_end - t_start) / (1.0 - t_start)
        sub_left, _ = _subdivide_3d(*right, s)
        return sub_left

    @staticmethod
    def _compute_kept_ranges(
        seg_start: float,
        seg_end: float,
        gap_regions: List[_TabRegion],
    ) -> Optional[List[Tuple[float, float]]]:
        overlapping = [
            (max(g_start, seg_start), min(g_end, seg_end))
            for g_start, g_end in gap_regions
            if g_start < seg_end and g_end > seg_start
        ]
        if not overlapping:
            return None
        kept = [(seg_start, seg_end)]
        for g_start, g_end in overlapping:
            new_kept = []
            for k_start, k_end in kept:
                if k_start < g_start:
                    new_kept.append((k_start, min(k_end, g_start)))
                if k_end > g_end:
                    new_kept.append((max(k_start, g_end), k_end))
            kept = new_kept
        return kept

    @staticmethod
    def _get_last_moving_end(
        cmds: List[Command],
    ) -> Optional[Point3D]:
        for c in reversed(cmds):
            if isinstance(c, MovingCommand) and c.end is not None:
                return c.end
        return None

    def _compute_hit_distance_original(
        self,
        commands: List[Command],
        target_geo_idx: int,
        target_t: float,
    ) -> Optional[float]:
        accum = 0.0
        last_pos: Optional[Point3D] = None
        geo_idx = 0

        for cmd in commands:
            if not isinstance(cmd, MovingCommand) or cmd.end is None:
                continue

            if isinstance(cmd, MoveToCommand):
                last_pos = cmd.end
                if geo_idx == target_geo_idx:
                    return accum
                geo_idx += 1
                continue

            if last_pos is None:
                last_pos = cmd.end
                geo_idx += 1
                continue

            if isinstance(cmd, BezierToCommand):
                seg_len = self._bezier_arc_length_2d(
                    last_pos, cmd.control1, cmd.control2, cmd.end
                )
            else:
                seg_len = math.dist(last_pos[:2], cmd.end[:2])

            if geo_idx == target_geo_idx:
                return accum + target_t * seg_len

            accum += seg_len
            last_pos = cmd.end
            geo_idx += 1

        return None

    def _compute_gap_regions_from_original(
        self,
        commands: List[Command],
        clip_data: List[_ClipPoint],
    ) -> List[_TabRegion]:
        temp_ops = Ops()
        temp_ops.replace_all(deepcopy(commands))
        temp_ops.preload_state()
        geo = temp_ops.to_geometry()

        if geo.data is None or len(geo.data) == 0:
            return []

        gap_regions: List[_TabRegion] = []
        for clip in clip_data:
            closest = geo.find_closest_point(clip.x, clip.y)
            if not closest:
                continue
            _, _, pt = closest
            dist_sq = (clip.x - pt[0]) ** 2 + (clip.y - pt[1]) ** 2
            if dist_sq > (clip.width * 2) ** 2:
                continue

            seg_idx, t, _ = closest
            hit_dist = self._compute_hit_distance_original(
                commands, seg_idx, t
            )
            if hit_dist is None:
                continue

            gap_regions.append(
                _TabRegion(
                    max(0.0, hit_dist - clip.width / 2.0),
                    hit_dist + clip.width / 2.0,
                )
            )

        return gap_regions

    def _clip_subpath_with_gaps(
        self,
        commands: List[Command],
        clip_data: List[_ClipPoint],
    ) -> List[Command]:
        gap_regions = self._compute_gap_regions_from_original(
            commands, clip_data
        )
        if not gap_regions:
            return commands

        new_cmds: List[Command] = []
        accum_dist = 0.0
        last_pos: Optional[Point3D] = None

        for cmd in commands:
            if isinstance(cmd, MoveToCommand):
                new_cmds.append(deepcopy(cmd))
                if cmd.end:
                    last_pos = cmd.end
                accum_dist = 0.0
                continue

            if not isinstance(cmd, MovingCommand):
                if not any(
                    g_start <= accum_dist <= g_end
                    for g_start, g_end in gap_regions
                ):
                    new_cmds.append(deepcopy(cmd))
                continue

            if cmd.end is None or last_pos is None:
                new_cmds.append(deepcopy(cmd))
                if cmd.end:
                    last_pos = cmd.end
                continue

            if isinstance(cmd, BezierToCommand):
                seg_len = self._bezier_arc_length_2d(
                    last_pos, cmd.control1, cmd.control2, cmd.end
                )
            else:
                seg_len = math.dist(last_pos[:2], cmd.end[:2])

            seg_start = accum_dist
            seg_end = accum_dist + seg_len

            if seg_len < 1e-9:
                last_pos = cmd.end
                accum_dist += seg_len
                continue

            kept = self._compute_kept_ranges(seg_start, seg_end, gap_regions)

            if kept is None:
                new_cmds.append(deepcopy(cmd))
            else:
                for k_start, k_end in kept:
                    d_start = k_start - seg_start
                    d_end = k_end - seg_start

                    if isinstance(cmd, BezierToCommand):
                        t_start = self._bezier_distance_to_t(
                            last_pos,
                            cmd.control1,
                            cmd.control2,
                            cmd.end,
                            d_start,
                        )
                        t_end = self._bezier_distance_to_t(
                            last_pos,
                            cmd.control1,
                            cmd.control2,
                            cmd.end,
                            d_end,
                        )
                        sub = self._extract_bezier_subsegment_3d(
                            last_pos,
                            cmd.control1,
                            cmd.control2,
                            cmd.end,
                            t_start,
                            t_end,
                        )

                        last_end = self._get_last_moving_end(new_cmds)
                        if (
                            last_end is not None
                            and math.dist(last_end[:2], sub[0][:2]) > 1e-6
                        ):
                            new_cmds.append(MoveToCommand(sub[0]))

                        new_cmd = BezierToCommand(
                            end=sub[3],
                            control1=sub[1],
                            control2=sub[2],
                        )
                        new_cmd.state = (
                            deepcopy(cmd.state) if cmd.state else None
                        )
                        new_cmds.append(new_cmd)
                    else:
                        t_s = d_start / seg_len
                        t_e = d_end / seg_len
                        start_pt = self._interpolate_point(
                            last_pos, cmd.end, t_s
                        )
                        end_pt = self._interpolate_point(
                            last_pos, cmd.end, t_e
                        )

                        last_end = self._get_last_moving_end(new_cmds)
                        if (
                            last_end is not None
                            and math.dist(last_end[:2], start_pt[:2]) > 1e-6
                        ):
                            new_cmds.append(MoveToCommand(start_pt))

                        new_line = LineToCommand(end_pt)
                        new_line.state = (
                            deepcopy(cmd.state) if cmd.state else None
                        )
                        new_cmds.append(new_line)

            accum_dist += seg_len
            last_pos = cmd.end

        orig_endpoint = None
        for c in reversed(commands):
            if isinstance(c, MovingCommand) and c.end:
                orig_endpoint = c.end
                break

        if orig_endpoint:
            last_end = self._get_last_moving_end(new_cmds)
            if last_end is None or math.dist(last_end, orig_endpoint) > 1e-6:
                new_cmds.append(MoveToCommand(orig_endpoint))

        return new_cmds

    def _insert_power_commands_curve_aware(
        self,
        commands: List[Command],
        clip_data: List[_ClipPoint],
        tab_power: float,
        original_power: float,
    ) -> List[Command]:
        tab_regions = self._compute_gap_regions_from_original(
            commands, clip_data
        )
        if not tab_regions:
            return commands
        tab_regions.sort(key=lambda r: r.start)

        new_cmds: List[Command] = []
        accum_dist = 0.0
        last_pos: Optional[Point3D] = None
        current_power = original_power

        for cmd in commands:
            if isinstance(cmd, MoveToCommand):
                new_cmds.append(deepcopy(cmd))
                if cmd.end:
                    last_pos = cmd.end
                accum_dist = 0.0
                continue

            if isinstance(cmd, SetPowerCommand):
                new_cmds.append(deepcopy(cmd))
                continue

            if not isinstance(cmd, MovingCommand):
                new_cmds.append(deepcopy(cmd))
                continue

            if cmd.end is None or last_pos is None:
                new_cmds.append(deepcopy(cmd))
                if cmd.end:
                    last_pos = cmd.end
                continue

            if isinstance(cmd, BezierToCommand):
                seg_len = self._bezier_arc_length_2d(
                    last_pos, cmd.control1, cmd.control2, cmd.end
                )
            else:
                seg_len = math.dist(last_pos[:2], cmd.end[:2])

            seg_start = accum_dist
            seg_end = accum_dist + seg_len

            if seg_len < 1e-9:
                last_pos = cmd.end
                accum_dist += seg_len
                continue

            events = self._collect_events(seg_start, seg_end, tab_regions)

            if not events:
                new_cmds.append(deepcopy(cmd))
            elif isinstance(cmd, BezierToCommand):
                self._split_bezier_with_power(
                    new_cmds,
                    cmd,
                    last_pos,
                    seg_start,
                    events,
                    tab_power,
                    original_power,
                    current_power,
                )
                current_power = self._get_final_power(
                    events, tab_power, original_power
                )
            else:
                for event_dist, event_type in events:
                    if event_dist > seg_start + 1e-9:
                        t = (event_dist - seg_start) / seg_len
                        split_pt = self._interpolate_point(
                            last_pos, cmd.end, t
                        )
                        new_line = LineToCommand(split_pt)
                        new_line.state = (
                            deepcopy(cmd.state) if cmd.state else None
                        )
                        new_cmds.append(new_line)

                    if event_type == _EventType.ENTER_TAB:
                        if current_power != tab_power:
                            new_cmds.append(SetPowerCommand(tab_power))
                            current_power = tab_power
                    else:
                        if current_power != original_power:
                            new_cmds.append(SetPowerCommand(original_power))
                            current_power = original_power

                new_cmds.append(deepcopy(cmd))

            accum_dist += seg_len
            last_pos = cmd.end

        return new_cmds

    def _split_bezier_with_power(
        self,
        new_cmds: List[Command],
        cmd: BezierToCommand,
        last_pos: Point3D,
        seg_start: float,
        events: List[Tuple[float, _EventType]],
        tab_power: float,
        original_power: float,
        current_power: float,
    ) -> None:
        p0 = last_pos
        c1, c2, p1 = cmd.control1, cmd.control2, cmd.end

        sub_segments: List[Tuple[float, float, float]] = []
        last_t = 0.0
        last_power = current_power

        for event_dist, event_type in events:
            d = event_dist - seg_start
            t_event = self._bezier_distance_to_t(p0, c1, c2, p1, d)
            if t_event > last_t + 1e-9:
                sub_segments.append((last_t, t_event, last_power))
            if event_type == _EventType.ENTER_TAB:
                last_power = tab_power
            else:
                last_power = original_power
            last_t = t_event

        if last_t < 1.0 - 1e-9:
            sub_segments.append((last_t, 1.0, last_power))

        for t_start, t_end, power in sub_segments:
            if power != current_power:
                new_cmds.append(SetPowerCommand(power))
                current_power = power

            sub = self._extract_bezier_subsegment_3d(
                p0, c1, c2, p1, t_start, t_end
            )
            new_cmd = BezierToCommand(
                end=sub[3], control1=sub[1], control2=sub[2]
            )
            new_cmd.state = deepcopy(cmd.state) if cmd.state else None
            new_cmds.append(new_cmd)

    @classmethod
    def from_dict(cls, data: Dict) -> "TabOpsTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(enabled=data.get("enabled", True))
