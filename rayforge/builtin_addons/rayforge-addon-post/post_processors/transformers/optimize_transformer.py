import logging
import math
from dataclasses import dataclass
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.state import State
from raygeo.ops.types import CommandCategory, CommandType
from scipy.spatial import cKDTree  # type: ignore

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo import Geometry


logger = logging.getLogger(__name__)


def _dist_2d(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Helper for 2D distance calculation on n-dimensional points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


@dataclass
class WorkpieceMeta:
    uid: str
    ops: Ops
    entry_point: Point3D
    exit_point: Point3D
    can_flip: bool = True


def _split_by_workpiece_markers(
    ops: Ops,
) -> List[Tuple[str, Ops]]:
    blocks: List[Tuple[str, Ops]] = []
    current_uid: Optional[str] = None
    current_block = Ops()

    for i in range(ops.len()):
        ct = ops.command_type(i)
        if ct == CommandType.WORKPIECE_START:
            current_uid = ops.workpiece_uid(i)
            current_block = Ops()
        elif ct == CommandType.WORKPIECE_END:
            if current_uid is not None:
                blocks.append((current_uid, current_block))
            current_uid = None
            current_block = Ops()
        elif ops.category(i) == CommandCategory.MOVING:
            if current_uid is not None:
                current_block.transfer_command_from(ops, i)

    if current_uid is not None and not current_block.is_empty():
        blocks.append((current_uid, current_block))

    return blocks


def _extract_workpiece_meta(uid: str, ops: Ops) -> Optional[WorkpieceMeta]:
    if ops.is_empty():
        return None

    entry_point: Optional[Point3D] = None
    exit_point: Optional[Point3D] = None

    for i in range(ops.len()):
        if ops.is_travel(i):
            entry_point = ops.endpoint(i)
            break

    if entry_point is None:
        for i in range(ops.len()):
            if ops.category(i) == CommandCategory.MOVING:
                entry_point = ops.endpoint(i)
                break

    for i in range(ops.len() - 1, -1, -1):
        if ops.category(i) == CommandCategory.MOVING:
            exit_point = ops.endpoint(i)
            break

    if entry_point is None or exit_point is None:
        return None

    can_flip = False
    for i in range(ops.len()):
        if ops.is_cutting(i):
            can_flip = True
            break

    return WorkpieceMeta(
        uid=uid,
        ops=ops,
        entry_point=entry_point,
        exit_point=exit_point,
        can_flip=can_flip,
    )


def _kdtree_order_workpieces(
    context: ProgressContext,
    metas: List[WorkpieceMeta],
    preserve_first: bool = False,
) -> List[WorkpieceMeta]:
    n = len(metas)
    if n < 2:
        return metas

    context.set_total(n)

    all_points = np.zeros((n * 2, 2))
    for i, meta in enumerate(metas):
        all_points[2 * i] = meta.entry_point[:2]
        all_points[2 * i + 1] = meta.exit_point[:2]

    kdtree = cKDTree(all_points)
    ordered: List[WorkpieceMeta] = []
    visited_mask = np.zeros(n, dtype=bool)

    if preserve_first:
        ordered.append(metas[0])
        visited_mask[0] = True
        current_pos = np.array(metas[0].exit_point[:2])
        context.set_progress(1)
    else:
        ordered.append(metas[0])
        visited_mask[0] = True
        current_pos = np.array(metas[0].exit_point[:2])
        context.set_progress(1)

    while len(ordered) < n:
        if context.is_cancelled():
            return ordered

        k = 10
        found_next = False

        while True:
            num_points_in_tree = kdtree.n
            query_k = min(k, num_points_in_tree)

            distances, indices = kdtree.query(current_pos, k=query_k)

            if not hasattr(indices, "__iter__"):
                indices = [indices]

            for point_idx in indices:
                segment_idx = point_idx // 2
                if not visited_mask[segment_idx]:
                    next_meta = metas[segment_idx]
                    is_exit_point = point_idx % 2 == 1

                    if next_meta.can_flip and is_exit_point:
                        next_meta = WorkpieceMeta(
                            uid=next_meta.uid,
                            ops=next_meta.ops.flip_ops(),
                            entry_point=next_meta.exit_point,
                            exit_point=next_meta.entry_point,
                            can_flip=next_meta.can_flip,
                        )
                        metas[segment_idx] = next_meta

                    ordered.append(next_meta)
                    visited_mask[segment_idx] = True
                    current_pos = np.array(next_meta.exit_point[:2])
                    found_next = True
                    break

            if found_next:
                break

            if k >= num_points_in_tree:
                logger.error(
                    "Workpiece optimizer could not find next workpiece."
                )
                for i in range(n):
                    if not visited_mask[i]:
                        ordered.append(metas[i])
                return ordered

            k *= 2

        context.set_progress(len(ordered))

    return ordered


def _two_opt_workpieces(
    context: ProgressContext,
    ordered: List[WorkpieceMeta],
    max_iter: int,
) -> List[WorkpieceMeta]:
    n = len(ordered)
    if n < 3:
        return ordered

    iter_count = 0
    improved = True
    context.set_total(max_iter)
    _hypot = math.hypot

    while improved and iter_count < max_iter:
        if context.is_cancelled():
            return ordered
        context.set_progress(iter_count)

        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if context.is_cancelled():
                    return ordered

                a_exit = ordered[i].exit_point
                b_entry = ordered[i + 1].entry_point
                e_exit = ordered[j].exit_point

                if j < n - 1:
                    f_entry = ordered[j + 1].entry_point

                    curr_cost = _hypot(
                        a_exit[0] - b_entry[0], a_exit[1] - b_entry[1]
                    ) + _hypot(e_exit[0] - f_entry[0], e_exit[1] - f_entry[1])
                    new_cost = _hypot(
                        a_exit[0] - e_exit[0], a_exit[1] - e_exit[1]
                    ) + _hypot(
                        b_entry[0] - f_entry[0], b_entry[1] - f_entry[1]
                    )
                else:
                    curr_cost = _hypot(
                        a_exit[0] - b_entry[0], a_exit[1] - b_entry[1]
                    )
                    new_cost = _hypot(
                        a_exit[0] - e_exit[0], a_exit[1] - e_exit[1]
                    )

                if new_cost < curr_cost:
                    sub = ordered[i + 1 : j + 1]
                    for k in range(len(sub)):
                        if sub[k].can_flip:
                            sub[k] = WorkpieceMeta(
                                uid=sub[k].uid,
                                ops=sub[k].ops.flip_ops(),
                                entry_point=sub[k].exit_point,
                                exit_point=sub[k].entry_point,
                                can_flip=sub[k].can_flip,
                            )
                    ordered[i + 1 : j + 1] = sub[::-1]
                    improved = True

        iter_count += 1

    context.set_progress(max_iter)
    return ordered


def _split_scanline(
    move_idx: int, scan_idx: int, source_ops: Ops
) -> List[Ops]:
    """
    Splits a single ScanLinePowerCommand into multiple segments if it
    contains areas of zero power (blank space). An overscanned line (with
    zero-power padding at the ends) is treated as a single segment.
    """
    pv = bytearray(source_ops.scanline_data(scan_idx))
    if not pv or not any(pv):
        return []

    stripped = pv.strip(b"\x00")
    if not stripped or 0 not in stripped:
        result = Ops()
        result.transfer_command_from(source_ops, move_idx)
        result.transfer_command_from(source_ops, scan_idx)
        return [result]

    is_on = np.array(pv) > 0
    padded = np.concatenate(([False], is_on, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    p_start = np.array(source_ops.endpoint(move_idx))
    p_end = np.array(source_ops.endpoint(scan_idx))
    line_vec = p_end - p_start
    num_steps = len(pv)

    state = source_ops.preloaded_state(scan_idx)

    segments = []
    for start_idx, end_idx in zip(starts, ends):
        t_start = start_idx / num_steps
        t_end = end_idx / num_steps

        seg_start_pt = tuple(p_start + t_start * line_vec)
        seg_end_pt = tuple(p_start + t_end * line_vec)
        power_slice = bytearray(pv[start_idx:end_idx])

        new_ops = Ops()
        new_ops.move_to(*seg_start_pt)
        new_ops.scan_to(*seg_end_pt, power_values=power_slice)
        new_ops.set_state_on_moving(state)

        segments.append(new_ops)
    return segments


def _group_paths_power_agnostic(
    ops: Ops,
) -> List[Ops]:
    """
    Groups commands into continuous path segments. This is used to
    handle zero-power LineTo commands created by transformers like Overscan.
    It defines a segment as a MoveTo followed by any number of non-travel
    moves, ignoring their power state for grouping purposes.
    """
    segments: List[Ops] = []
    if ops.is_empty():
        return []
    i = 0
    while i < ops.len():
        if not ops.is_travel(i):
            i += 1
            continue
        current_segment = Ops()
        current_segment.transfer_command_from(ops, i)
        i += 1
        # Consume all subsequent drawing commands (LineTo, ArcTo) regardless
        # of power.
        while i < ops.len() and not ops.is_travel(i):
            current_segment.transfer_command_from(ops, i)
            i += 1
        segments.append(current_segment)
    return segments


def group_mixed_continuity(
    ops: Ops,
) -> List[Ops]:
    """
    Splits a command list into continuous path segments. It correctly pairs
    a MoveTo command with a subsequent ScanLinePowerCommand, splitting it if
    necessary, to form optimizable raster segments.
    """
    segments: List[Ops] = []
    if ops.is_empty():
        return []

    i = 0
    while i < ops.len():
        # A segment must start with a travel command (MoveTo).
        if not ops.is_travel(i):
            # This handles malformed lists or finds the next MoveTo.
            i += 1
            continue

        # Check what follows the travel move
        if (i + 1) < ops.len() and ops.is_scanline(i + 1):
            sub_segments = _split_scanline(i, i + 1, ops)
            if sub_segments:
                segments.extend(sub_segments)
            i += 2
        else:
            # Fallback to power-agnostic grouping for vector paths.
            current_segment = Ops()
            current_segment.transfer_command_from(ops, i)
            i += 1
            while i < ops.len() and not ops.is_travel(i):
                # Defensively handle mixed vector/raster types
                if ops.is_scanline(i):
                    break
                current_segment.transfer_command_from(ops, i)
                i += 1
            segments.append(current_segment)

    return segments


def kdtree_order_segments(
    context: ProgressContext, segments: List[Ops]
) -> List[Ops]:
    """
    Orders segments using a nearest-neighbor search accelerated by a k-d tree.
    This provides a fast and robust O(N log N) implementation.
    """
    n = len(segments)
    if n < 2:
        return segments

    context.set_total(n)

    # 1. Build the "geographic map" (k-d tree).
    # We create a list of all start/end points. Point 2*i is the start of
    # segment i, and point 2*i+1 is the end.
    all_points = np.zeros((n * 2, 2))
    for i, seg in enumerate(segments):
        all_points[2 * i] = seg.endpoint(0)[:2]
        all_points[2 * i + 1] = seg.endpoint(seg.len() - 1)[:2]

    kdtree = cKDTree(all_points)
    ordered_segments: List[Ops] = []
    visited_mask = np.zeros(n, dtype=bool)

    # 2. Pick a starting point and initialize.
    current_segment_idx = 0
    current_seg = segments[current_segment_idx]
    ordered_segments.append(current_seg)
    visited_mask[current_segment_idx] = True
    current_pos = np.array(current_seg.endpoint(current_seg.len() - 1)[:2])
    context.set_progress(1)

    # 3. Iteratively find the closest unvisited segment.
    while len(ordered_segments) < n:
        if context.is_cancelled():
            return ordered_segments

        # Query for several neighbors. k must be large enough to find an
        # unvisited point, even if the closest points belong to an already
        # visited segment. A small constant is a good heuristic.
        # Scipy's cKDTree handles k > number of points gracefully.
        k = 10
        found_next = False

        # Retry loop for finding the next segment with dynamic k
        while True:
            num_points_in_tree = kdtree.n
            query_k = min(k, num_points_in_tree)

            distances, indices = kdtree.query(current_pos, k=query_k)

            if not hasattr(indices, "__iter__"):
                indices = [indices]

            for point_idx in indices:
                segment_idx = point_idx // 2
                if not visited_mask[segment_idx]:
                    # This is our next segment.
                    next_seg = segments[segment_idx]
                    is_end_point = point_idx % 2 == 1

                    if is_end_point:
                        next_seg = next_seg.flip_ops()

                    ordered_segments.append(next_seg)
                    visited_mask[segment_idx] = True
                    current_pos = np.array(
                        next_seg.endpoint(next_seg.len() - 1)[:2]
                    )
                    found_next = True
                    break  # break `for point_idx` loop

            if found_next:
                break  # break `while True` retry loop

            # If not found, check if we can expand the search
            if k >= num_points_in_tree:
                # Can't expand search, this is the error condition
                logger.error("Path optimizer could not find a next segment.")
                # Add remaining segments to avoid losing paths, though they
                # won't be ordered
                for i in range(n):
                    if not visited_mask[i]:
                        ordered_segments.append(segments[i])
                return ordered_segments  # Exit function

            # Expand search and retry
            k *= 2

        context.set_progress(len(ordered_segments))

    return ordered_segments


def greedy_order_segments(
    context: ProgressContext,
    segments: List[Ops],
) -> List[Ops]:
    """
    Greedy ordering using vectorized math.dist computations.
    O(N^2) complexity.

    It is assumed that the input segments contain only Ops objects
    with moving commands, so it is ensured that each command has
    x,y coordinates.
    """
    if not segments:
        return []

    # Make a shallow copy of the list so we can pop from it
    remaining = list(segments)

    context.set_total(len(remaining))
    ordered: List[Ops] = []

    # Take the first segment as is
    current_seg = remaining.pop(0)
    ordered.append(current_seg)
    current_pos = np.array(current_seg.endpoint(current_seg.len() - 1))
    context.set_progress(1)

    while remaining:
        if context.is_cancelled():
            return ordered

        # Vectorized distance calculation to all start and end points
        starts = np.array([seg.endpoint(0) for seg in remaining])
        ends = np.array([seg.endpoint(seg.len() - 1) for seg in remaining])

        d_starts = np.linalg.norm(starts[:, :2] - current_pos[:2], axis=1)
        d_ends = np.linalg.norm(ends[:, :2] - current_pos[:2], axis=1)

        # Find the minimum distance for each segment (start or end)
        candidate_dists = np.minimum(d_starts, d_ends)
        best_idx = int(np.argmin(candidate_dists))

        best_seg = remaining.pop(best_idx)

        # If the end was closer, flip the segment
        if d_ends[best_idx] < d_starts[best_idx]:
            best_seg = best_seg.flip_ops()

        ordered.append(best_seg)
        current_pos = np.array(best_seg.endpoint(best_seg.len() - 1))
        context.set_progress(len(ordered))

    return ordered


def two_opt(
    context: ProgressContext,
    ordered: List[Ops],
    max_iter: int,
) -> List[Ops]:
    """
    2-opt: try reversing entire sub-sequences if that lowers the travel cost.
    """
    n = len(ordered)
    if n < 3:
        return ordered

    iter_count = 0
    improved = True
    context.set_total(max_iter)
    _hypot = math.hypot

    while improved and iter_count < max_iter:
        if context.is_cancelled():
            return ordered
        context.set_progress(iter_count)

        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if context.is_cancelled():
                    return ordered

                a_end = ordered[i].endpoint(ordered[i].len() - 1)
                b_start = ordered[i + 1].endpoint(0)
                e_end = ordered[j].endpoint(ordered[j].len() - 1)

                if j < n - 1:
                    f_start = ordered[j + 1].endpoint(0)

                    curr_cost = _hypot(
                        a_end[0] - b_start[0], a_end[1] - b_start[1]
                    ) + _hypot(e_end[0] - f_start[0], e_end[1] - f_start[1])
                    new_cost = _hypot(
                        a_end[0] - e_end[0], a_end[1] - e_end[1]
                    ) + _hypot(
                        b_start[0] - f_start[0], b_start[1] - f_start[1]
                    )
                else:
                    curr_cost = _hypot(
                        a_end[0] - b_start[0], a_end[1] - b_start[1]
                    )
                    new_cost = _hypot(a_end[0] - e_end[0], a_end[1] - e_end[1])

                if new_cost < curr_cost:
                    # Decision made. Now perform the mutation.
                    sub = ordered[i + 1 : j + 1]
                    # Reverse order and flip each segment.
                    for k in range(len(sub)):
                        sub[k] = sub[k].flip_ops()
                    ordered[i + 1 : j + 1] = sub[::-1]
                    improved = True
        iter_count += 1

    context.set_progress(max_iter)
    return ordered


def _prepare_optimization_jobs(
    long_segments: List[Ops],
    two_opt_segment_threshold: int,
    two_opt_command_limit: int,
) -> List[Dict[str, Any]]:
    """
    Categorizes long_segments into jobs for optimization based on their
    complexity.

    This function implements the bucketing logic:
    1. Segments are identified as passthrough (e.g., markers or single-path
       segments), too large for 2-opt, or candidates for 2-opt.
    2. 2-opt candidates are sorted by their number of sub-segments.
    3. The smallest candidates are placed into a "bucket" for 2-opt refinement
       until the bucket's total command count is reached.
    4. Candidates that don't fit in the bucket are downgraded to k-d tree only.

    Returns a list of job dictionaries, each representing one long_segment.
    """
    jobs = []
    two_opt_candidates = []

    for i, long_segment in enumerate(long_segments):
        # Handle passthrough segments like markers
        if long_segment.is_empty() or long_segment.is_marker(0):
            jobs.append(
                {
                    "type": "passthrough",
                    "original_index": i,
                    "workload": 1,
                    "original_segment": long_segment,
                }
            )
            continue

        contains_scanline = any(
            long_segment.is_scanline(j) for j in range(long_segment.len())
        )
        # Split the long segment into its reorderable sub-segments
        if contains_scanline:
            sub_segments = group_mixed_continuity(long_segment)
        else:
            sub_segments = _group_paths_power_agnostic(long_segment)

        num_sub_segments = len(sub_segments)

        # If there is nothing to reorder (0 or 1 path), treat it as a
        # passthrough job to avoid unnecessary processing overhead.
        if num_sub_segments <= 1:
            jobs.append(
                {
                    "type": "passthrough",
                    "original_index": i,
                    "workload": 1,
                    "original_segment": long_segment,
                }
            )
            continue

        # Categorize: large segments go directly to kdtree_only jobs
        if num_sub_segments > two_opt_segment_threshold:
            jobs.append(
                {
                    "type": "kdtree_only",
                    "original_index": i,
                    "workload": num_sub_segments,
                    "sub_segments": sub_segments,
                }
            )
        else:
            # Otherwise, it's a candidate for 2-opt
            command_count = sum(seg.len() for seg in sub_segments)
            two_opt_candidates.append(
                {
                    "original_index": i,
                    "workload": num_sub_segments,
                    "sub_segments": sub_segments,
                    "command_count": command_count,
                }
            )

    # Sort candidates by size (smallest first) to prioritize them for 2-opt
    two_opt_candidates.sort(key=lambda c: c["workload"])

    # Fill the 2-opt bucket based on the command limit
    bucketed_command_count = 0
    for candidate in two_opt_candidates:
        if (
            bucketed_command_count + candidate["command_count"]
            <= two_opt_command_limit
        ):
            jobs.append(
                {
                    "type": "two_opt",
                    "original_index": candidate["original_index"],
                    "workload": candidate["workload"],
                    "sub_segments": candidate["sub_segments"],
                }
            )
            bucketed_command_count += candidate["command_count"]
        else:
            # Candidate did not fit, downgrade to kdtree_only
            jobs.append(
                {
                    "type": "kdtree_only",
                    "original_index": candidate["original_index"],
                    "workload": candidate["workload"],
                    "sub_segments": candidate["sub_segments"],
                }
            )

    return jobs


_DEFAULT_STATE = State(
    power=0.0,
    air_assist=False,
    cut_speed=None,
    travel_speed=None,
    active_laser_uid=None,
    frequency=None,
    pulse_width=None,
)


def _sync_state_commands(
    ops: Ops,
    state: State,
    prev: State,
) -> State:
    """Emits state commands on ops for fields that differ from prev.

    Returns the updated prev state.
    """
    if state.power != prev.power:
        ops.set_power(state.power)
        prev.power = state.power
    if state.cut_speed is not None and state.cut_speed != prev.cut_speed:
        ops.set_cut_speed(state.cut_speed)
        prev.cut_speed = state.cut_speed
    if (
        state.travel_speed is not None
        and state.travel_speed != prev.travel_speed
    ):
        ops.set_travel_speed(state.travel_speed)
        prev.travel_speed = state.travel_speed
    if state.air_assist != prev.air_assist:
        if state.air_assist:
            ops.enable_air_assist()
        else:
            ops.disable_air_assist()
        prev.air_assist = state.air_assist
    if (
        state.active_laser_uid is not None
        and state.active_laser_uid != prev.active_laser_uid
    ):
        ops.set_laser(state.active_laser_uid)
        prev.active_laser_uid = state.active_laser_uid
    return prev


class Optimize(OpsTransformer):
    """
    Optimizes toolpaths to minimize travel distance using a hybrid approach.

    Performs two levels of optimization:
    1. Workpiece-level: Reorders and flips workpieces to minimize
       inter-workpiece travel (when multiple workpieces are present).
    2. Segment-level: Reorders path segments within each workpiece.

    For segment optimization, it categorizes path segments and applies
    optimization strategies accordingly:
    1. A fast k-d tree nearest-neighbor search is applied to all segments for
       a good initial ordering.
    2. For segments with fewer paths than a threshold, a more intensive 2-opt
       refinement is considered.
    3. A "bucket" of the smallest of these candidate segments is created, up
       to a total command limit, to receive 2-opt refinement. This focuses
       the most expensive optimization where it is most effective and keeps
       the total runtime predictable.

    The process is:
    1. Preprocess the command list to attach state (power, speed, etc.) to
       each moving command.
    2. Group commands into continuous, non-reorderable `long_segments`.
    3. Categorize and bucket these segments into optimization jobs.
    4. Execute optimization on each job according to its type.
    5. Re-assemble the final command list with state commands re-inserted.
    """

    def __init__(
        self,
        enabled: bool = True,
        allow_flip: bool = True,
        preserve_first: bool = False,
        preserve_order: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(enabled=enabled, **kwargs)
        self.allow_flip = allow_flip
        self.preserve_first = preserve_first
        self.preserve_order = preserve_order or []

    @property
    def execution_phase(self) -> ExecutionPhase:
        """Path optimization should run last on the final path segments."""
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Optimize Path")

    @property
    def description(self) -> str:
        return _("Minimizes travel distance by reordering segments.")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if context is None:
            return

        ops.preload_state()
        if context.is_cancelled():
            return

        if workpiece is None:
            blocks = _split_by_workpiece_markers(ops)
            if len(blocks) >= 2:
                self._optimize_workpiece_order(ops, blocks, context)
                return

        self._optimize_segments(ops, context)

    def _optimize_workpiece_order(
        self,
        ops: Ops,
        blocks: List[Tuple[str, Ops]],
        context: ProgressContext,
    ) -> None:
        context.set_message(_("Analyzing workpieces..."))

        metas: List[WorkpieceMeta] = []
        for uid, block_ops in blocks:
            meta = _extract_workpiece_meta(uid, block_ops)
            if meta is not None:
                if not self.allow_flip:
                    meta = WorkpieceMeta(
                        uid=meta.uid,
                        ops=meta.ops,
                        entry_point=meta.entry_point,
                        exit_point=meta.exit_point,
                        can_flip=False,
                    )
                metas.append(meta)

        if len(metas) < 2:
            return

        preserved_indices: set = set()
        reorderable_metas: List[WorkpieceMeta] = []

        for i, meta in enumerate(metas):
            if meta.uid in self.preserve_order or (
                self.preserve_first and i == 0
            ):
                preserved_indices.add(i)
            else:
                reorderable_metas.append(meta)

        if not reorderable_metas:
            return

        context.set_message(_("Optimizing workpiece order..."))

        kdtree_weight = 0.7
        kdtree_ctx = context.sub_context(
            base_progress=0.0, progress_range=kdtree_weight, total=1.0
        )
        ordered_metas = _kdtree_order_workpieces(
            kdtree_ctx, reorderable_metas, preserve_first=False
        )

        two_opt_ctx = context.sub_context(
            base_progress=kdtree_weight,
            progress_range=0.3,
            total=1.0,
        )
        context.set_message(_("Applying 2-opt refinement..."))
        ordered_metas = _two_opt_workpieces(two_opt_ctx, ordered_metas, 10)

        context.set_message(_("Reassembling optimized workpieces..."))

        if preserved_indices:
            final_metas: List[WorkpieceMeta] = []
            reorder_idx = 0
            for i in range(len(metas)):
                if i in preserved_indices:
                    final_metas.append(metas[i])
                else:
                    if reorder_idx < len(ordered_metas):
                        final_metas.append(ordered_metas[reorder_idx])
                        reorder_idx += 1
            ordered_metas = final_metas

        self._reassemble_workpieces(ops, ordered_metas, context)

        logger.debug(
            f"Workpiece optimization finished: {len(ordered_metas)} "
            "workpieces reordered"
        )
        context.set_message(_("Workpiece optimization complete"))
        context.set_progress(1.0)
        context.flush()

    def _reassemble_workpieces(
        self,
        ops: Ops,
        ordered_metas: List[WorkpieceMeta],
        context: ProgressContext,
    ) -> None:
        ops.preload_state()

        ops.clear()

        prev = _DEFAULT_STATE
        for meta in ordered_metas:
            ops.workpiece_start(meta.uid)

            for j in range(meta.ops.len()):
                state = meta.ops.preloaded_state(j)
                prev = _sync_state_commands(ops, state, prev)
                ops.transfer_command_from(meta.ops, j)

            ops.workpiece_end(meta.uid)

    def _optimize_segments(self, ops: Ops, context: ProgressContext) -> None:
        # Thresholds for the smart optimization strategy
        TWO_OPT_SEGMENT_THRESHOLD = 1000
        TWO_OPT_COMMAND_LIMIT = 10000

        # Step 1: Preprocessing
        context.set_message(_("Preprocessing for optimization..."))

        nons = ops.without_state()
        logger.debug(f"Optimizing {nons.len()} moving commands.")

        # Step 2: Splitting into non-reorderable long segments
        long_segments = nons.group_by_state_continuity()
        if context.is_cancelled():
            return

        # Define weights for the progress reporting of the main
        # optimization loop vs final reassembly.
        optimize_weight = 0.9
        reassemble_weight = 0.1

        # This context covers the main optimization loop over all
        # long_segments.
        optimize_ctx = context.sub_context(
            base_progress=0.0, progress_range=optimize_weight, total=1.0
        )

        # Step 3: Categorize and bucket segments into optimization jobs
        context.set_message(_("Analyzing and bucketing path segments..."))
        jobs = _prepare_optimization_jobs(
            long_segments, TWO_OPT_SEGMENT_THRESHOLD, TWO_OPT_COMMAND_LIMIT
        )

        # Pre-calculate total workload for a smooth progress bar
        total_workload = sum(job.get("workload", 1) for job in jobs)
        cumulative_workload = 0.0
        processed_results: Dict[int, Any] = {}

        # Step 4: Execute optimization jobs
        for i, job in enumerate(jobs):
            if context.is_cancelled():
                return

            current_workload = job.get("workload", 1)
            progress_range = (
                current_workload / total_workload if total_workload > 0 else 0
            )
            base_progress = (
                cumulative_workload / total_workload
                if total_workload > 0
                else 0
            )
            segment_ctx = optimize_ctx.sub_context(
                base_progress=base_progress,
                progress_range=progress_range,
                total=1.0,
            )
            context.set_message(
                _("Optimizing segment {i}/{total}...").format(
                    i=i + 1, total=len(jobs)
                )
            )

            job_type = job["type"]
            if job_type == "passthrough":
                # For markers or non-reorderable segments, just pass through.
                processed_results[job["original_index"]] = job[
                    "original_segment"
                ]

            elif job_type in ("kdtree_only", "two_opt"):
                sub_segments = job["sub_segments"]

                # All optimizable jobs start with k-d tree
                kdtree_weight = 0.7 if job_type == "two_opt" else 1.0
                kdtree_ctx = segment_ctx.sub_context(
                    base_progress=0.0, progress_range=kdtree_weight, total=1.0
                )
                segment_ctx.set_message(_("Finding nearest paths..."))
                ordered_segments = kdtree_order_segments(
                    kdtree_ctx, sub_segments
                )

                final_segments = ordered_segments
                if job_type == "two_opt":
                    logger.info(
                        f"Segment {job['original_index']} is small "
                        f"({len(sub_segments)} sub-segments), "
                        "applying 2-opt refinement."
                    )
                    two_opt_ctx = segment_ctx.sub_context(
                        base_progress=kdtree_weight,
                        progress_range=0.3,
                        total=1.0,
                    )
                    segment_ctx.set_message(_("Applying 2-opt refinement..."))
                    final_segments = two_opt(two_opt_ctx, ordered_segments, 10)

                processed_results[job["original_index"]] = final_segments

            cumulative_workload += current_workload

        # Ensure the optimization part reports full completion.
        optimize_ctx.set_progress(1.0)

        # Step 5: Re-assemble the Ops object from processed results.
        context.set_message(_("Reassembling optimized paths..."))
        reassemble_ctx = context.sub_context(
            base_progress=optimize_weight,
            progress_range=reassemble_weight,
            total=1.0,
        )

        # Reconstruct the result in the original order of long_segments
        result = [
            processed_results[i]
            for i in range(len(long_segments))
            if i in processed_results
        ]

        flat_result_segments: List[Ops] = []
        for item in result:
            if isinstance(item, list):
                flat_result_segments.extend(item)
            else:
                flat_result_segments.append(item)

        reassemble_ctx.set_total(len(flat_result_segments))
        ops.clear()
        prev = _DEFAULT_STATE
        for i, segment_ops in enumerate(flat_result_segments):
            if segment_ops.is_empty():
                continue

            if segment_ops.is_marker(0):
                ops.transfer_command_from(segment_ops, 0)
                continue

            for j in range(segment_ops.len()):
                state = segment_ops.preloaded_state(j)
                prev = _sync_state_commands(ops, state, prev)
                ops.transfer_command_from(segment_ops, j)
            reassemble_ctx.set_progress(i + 1)

        logger.debug("Optimization finished")
        context.set_message(_("Optimization complete"))
        context.set_progress(1.0)
        context.flush()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        result = super().to_dict()
        result["allow_flip"] = self.allow_flip
        result["preserve_first"] = self.preserve_first
        result["preserve_order"] = self.preserve_order
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Optimize":
        """Creates an Optimize instance from a dictionary."""
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            allow_flip=data.get("allow_flip", True),
            preserve_first=data.get("preserve_first", False),
            preserve_order=data.get("preserve_order", []),
        )
