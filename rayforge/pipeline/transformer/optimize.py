import numpy as np
import math
import logging
from typing import Optional, List, Dict, Any, Tuple, cast
from scipy.spatial import cKDTree  # type: ignore
from ...core.workpiece import WorkPiece
from ...core.ops import (
    Ops,
    State,
    MovingCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    Command,
)
from ...core.ops.flip import flip_segment
from ...core.ops.group import (
    group_by_state_continuity,
)
from .base import OpsTransformer, ExecutionPhase
from ...shared.tasker.context import BaseExecutionContext, ExecutionContext


logger = logging.getLogger(__name__)


def _dist_2d(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Helper for 2D distance calculation on n-dimensional points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _split_scanline(
    move_cmd: MovingCommand, scan_cmd: ScanLinePowerCommand
) -> List[List[MovingCommand]]:
    """
    Splits a single ScanLinePowerCommand into multiple segments if it
    contains areas of zero power (blank space). An overscanned line (with
    zero-power padding at the ends) is treated as a single segment.
    """
    if not scan_cmd.power_values or not np.any(scan_cmd.power_values):
        return []

    # Find contiguous "on" segments.
    is_on = np.array(scan_cmd.power_values) > 0
    padded = np.concatenate(([False], is_on, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]

    # If there's only one "on" segment, it's either a fully "on" line or an
    # overscanned line. In either case, we treat it as a single,
    # non-splittable segment.
    if len(starts) <= 1:
        # Note: len(starts) == 0 is covered by the np.any() check above,
        # but this condition is safer.
        return [[move_cmd, scan_cmd]]

    # If we reach here, there are multiple segments that need to be created.
    ends = np.where(diffs == -1)[0]

    p_start = np.array(move_cmd.end)
    p_end = np.array(scan_cmd.end)
    line_vec = p_end - p_start
    num_steps = len(scan_cmd.power_values)

    segments = []
    for start_idx, end_idx in zip(starts, ends):
        t_start = start_idx / num_steps
        t_end = end_idx / num_steps

        seg_start_pt = p_start + t_start * line_vec
        seg_end_pt = p_start + t_end * line_vec
        power_slice = scan_cmd.power_values[start_idx:end_idx]

        new_move = MoveToCommand(tuple(seg_start_pt))
        new_scan = ScanLinePowerCommand(
            tuple(seg_end_pt), power_values=power_slice
        )

        if scan_cmd.state:
            new_move.state = scan_cmd.state
            new_scan.state = scan_cmd.state

        segments.append([new_move, new_scan])
    return segments


def _group_paths_power_agnostic(
    commands: List[MovingCommand],
) -> List[List[MovingCommand]]:
    """
    Groups commands into continuous path segments. This is used to
    handle zero-power LineTo commands created by transformers like Overscan.
    It defines a segment as a MoveTo followed by any number of non-travel
    moves, ignoring their power state for grouping purposes.
    """
    segments: List[List[MovingCommand]] = []
    if not commands:
        return []
    i = 0
    while i < len(commands):
        start_cmd = commands[i]
        if not start_cmd.is_travel_command():
            i += 1
            continue
        current_segment = [start_cmd]
        i += 1
        # Consume all subsequent drawing commands (LineTo, ArcTo) regardless
        # of power.
        while i < len(commands) and not commands[i].is_travel_command():
            current_segment.append(commands[i])
            i += 1
        segments.append(current_segment)
    return segments


def group_mixed_continuity(
    commands: List[MovingCommand],
) -> List[List[MovingCommand]]:
    """
    Splits a command list into continuous path segments. It correctly pairs
    a MoveTo command with a subsequent ScanLinePowerCommand, splitting it if
    necessary, to form optimizable raster segments.
    """
    segments: List[List[MovingCommand]] = []
    if not commands:
        return []

    i = 0
    while i < len(commands):
        # A segment must start with a travel command (MoveTo).
        start_cmd = commands[i]
        if not start_cmd.is_travel_command():
            # This handles malformed lists or finds the next MoveTo.
            i += 1
            continue

        # Check what follows the travel move
        if (i + 1) < len(commands) and isinstance(
            commands[i + 1], ScanLinePowerCommand
        ):
            move = start_cmd
            scan = cast(ScanLinePowerCommand, commands[i + 1])
            sub_segments = _split_scanline(move, scan)
            if sub_segments:
                segments.extend(sub_segments)
            i += 2  # Consume both commands
        else:
            # Fallback to power-agnostic grouping for vector paths.
            current_segment = [start_cmd]
            i += 1
            while i < len(commands) and not commands[i].is_travel_command():
                # Defensively handle mixed vector/raster types
                if isinstance(commands[i], ScanLinePowerCommand):
                    break
                current_segment.append(commands[i])
                i += 1
            segments.append(current_segment)

    return segments


def kdtree_order_segments(
    context: BaseExecutionContext, segments: List[List[MovingCommand]]
) -> List[List[MovingCommand]]:
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
        all_points[2 * i] = seg[0].end[:2]
        all_points[2 * i + 1] = seg[-1].end[:2]

    kdtree = cKDTree(all_points)
    ordered_segments = []
    visited_mask = np.zeros(n, dtype=bool)

    # 2. Pick a starting point and initialize.
    current_segment_idx = 0
    current_seg = segments[current_segment_idx]
    ordered_segments.append(current_seg)
    visited_mask[current_segment_idx] = True
    current_pos = np.array(current_seg[-1].end[:2])
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
                        next_seg = flip_segment(next_seg)

                    ordered_segments.append(next_seg)
                    visited_mask[segment_idx] = True
                    current_pos = np.array(next_seg[-1].end[:2])
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
    context: BaseExecutionContext,
    segments: List[List[MovingCommand]],
) -> List[List[MovingCommand]]:
    """
    Greedy ordering using vectorized math.dist computations.
    O(N^2) complexity.

    It is assumed that the input segments contain only Command objects
    that are NOT state commands (such as 'set_power'), so it is
    ensured that each Command performs a position change (i.e. it has
    x,y coordinates).
    """
    if not segments:
        return []

    # Make a shallow copy of the list so we can pop from it
    remaining = list(segments)

    context.set_total(len(remaining))
    ordered: List[List[MovingCommand]] = []

    # Take the first segment as is
    current_seg = remaining.pop(0)
    ordered.append(current_seg)
    current_pos = np.array(current_seg[-1].end)
    context.set_progress(1)

    while remaining:
        if context.is_cancelled():
            return ordered

        # Vectorized distance calculation to all start and end points
        starts = np.array([seg[0].end for seg in remaining])
        ends = np.array([seg[-1].end for seg in remaining])

        d_starts = np.linalg.norm(starts[:, :2] - current_pos[:2], axis=1)
        d_ends = np.linalg.norm(ends[:, :2] - current_pos[:2], axis=1)

        # Find the minimum distance for each segment (start or end)
        candidate_dists = np.minimum(d_starts, d_ends)
        best_idx = int(np.argmin(candidate_dists))

        best_seg = remaining.pop(best_idx)

        # If the end was closer, flip the segment
        if d_ends[best_idx] < d_starts[best_idx]:
            best_seg = flip_segment(best_seg)

        ordered.append(best_seg)
        current_pos = np.array(best_seg[-1].end)
        context.set_progress(len(ordered))

    return ordered


def two_opt(
    context: BaseExecutionContext,
    ordered: List[List[MovingCommand]],
    max_iter: int,
) -> List[List[MovingCommand]]:
    """
    2-opt: try reversing entire sub-sequences if that lowers the travel cost.
    """
    n = len(ordered)
    if n < 3:
        return ordered

    iter_count = 0
    improved = True
    context.set_total(max_iter)

    while improved and iter_count < max_iter:
        if context.is_cancelled():
            return ordered
        context.set_progress(iter_count)

        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if context.is_cancelled():
                    return ordered

                a_end = ordered[i][-1].end
                b_start = ordered[i + 1][0].end
                e_end = ordered[j][-1].end

                if j < n - 1:
                    f_start = ordered[j + 1][0].end

                    # Current path:
                    # ... -> A_end -> B_start -> ... -> E_end -> F_start -> ...
                    curr_cost = _dist_2d(a_end, b_start) + _dist_2d(
                        e_end, f_start
                    )

                    # New path:
                    # ... -> A_end -> E_end -> ... -> B_start -> F_start -> ...
                    new_cost = _dist_2d(a_end, e_end) + _dist_2d(
                        b_start, f_start
                    )
                else:
                    # J is the last segment
                    curr_cost = _dist_2d(a_end, b_start)
                    new_cost = _dist_2d(a_end, e_end)

                if new_cost < curr_cost:
                    # Decision made. Now perform the mutation.
                    sub = ordered[i + 1 : j + 1]
                    # Reverse order and flip each segment.
                    for k in range(len(sub)):
                        sub[k] = flip_segment(sub[k])
                    ordered[i + 1 : j + 1] = sub[::-1]
                    improved = True
        iter_count += 1

    context.set_progress(max_iter)
    return ordered


def _prepare_optimization_jobs(
    long_segments: List[List[Command]],
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
        if not long_segment or long_segment[0].is_marker():
            jobs.append(
                {
                    "type": "passthrough",
                    "original_index": i,
                    "workload": 1,
                    "original_segment": long_segment,
                }
            )
            continue

        # Split the long segment into its reorderable sub-segments
        contains_scanline = any(
            isinstance(c, ScanLinePowerCommand) for c in long_segment
        )
        if contains_scanline:
            sub_segments = group_mixed_continuity(
                cast(List[MovingCommand], long_segment)
            )
        else:
            sub_segments = _group_paths_power_agnostic(
                cast(List[MovingCommand], long_segment)
            )

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
            command_count = sum(len(s) for s in sub_segments)
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


class Optimize(OpsTransformer):
    """
    Optimizes toolpaths to minimize travel distance using a hybrid approach.

    It categorizes path segments and applies optimization strategies
    accordingly:
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
        context: Optional[BaseExecutionContext] = None,
    ) -> None:
        if context is None:
            context = ExecutionContext()

        # Thresholds for the smart optimization strategy
        TWO_OPT_SEGMENT_THRESHOLD = 1000
        TWO_OPT_COMMAND_LIMIT = 10000

        # Step 1: Preprocessing
        context.set_message(_("Preprocessing for optimization..."))
        ops.preload_state()
        if context.is_cancelled():
            return

        commands = [c for c in ops if not c.is_state_command()]
        logger.debug(f"Optimizing {len(commands)} moving commands.")

        # Step 2: Splitting into non-reorderable long segments
        long_segments = group_by_state_continuity(commands)
        if context.is_cancelled():
            return

        # Define weights for the progress reporting of the main
        # optimization loop vs final reassembly.
        optimize_weight = 0.9
        reassemble_weight = 0.1

        # This context covers the main optimization loop over all
        # long_segments.
        optimize_ctx = context.sub_context(
            base_progress=0.0, progress_range=optimize_weight
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
                base_progress=base_progress, progress_range=progress_range
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
                    base_progress=0.0, progress_range=kdtree_weight
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
                        base_progress=kdtree_weight, progress_range=0.3
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
            base_progress=optimize_weight, progress_range=reassemble_weight
        )

        # Reconstruct the result in the original order of long_segments
        result = [
            processed_results[i]
            for i in range(len(long_segments))
            if i in processed_results
        ]

        flat_result_segments = []
        for item in result:
            if item and isinstance(item[0], list):
                flat_result_segments.extend(item)
            else:
                flat_result_segments.append(item)

        reassemble_ctx.set_total(len(flat_result_segments))
        ops.clear()
        prev_state = State()
        for i, segment in enumerate(flat_result_segments):
            if not segment:
                continue

            if segment[0].is_marker():
                ops.add(segment[0])
                continue

            for cmd in segment:
                if cmd.state.air_assist != prev_state.air_assist:
                    ops.enable_air_assist(cmd.state.air_assist)
                    prev_state.air_assist = cmd.state.air_assist
                if cmd.state.power != prev_state.power:
                    ops.set_power(cmd.state.power)
                    prev_state.power = cmd.state.power
                if cmd.state.cut_speed != prev_state.cut_speed:
                    ops.set_cut_speed(cmd.state.cut_speed)
                    prev_state.cut_speed = cmd.state.cut_speed
                if cmd.state.travel_speed != prev_state.travel_speed:
                    ops.set_travel_speed(cmd.state.travel_speed)
                    prev_state.travel_speed = cmd.state.travel_speed
                if cmd.state.active_laser_uid != prev_state.active_laser_uid:
                    ops.set_laser(cmd.state.active_laser_uid)
                    prev_state.active_laser_uid = cmd.state.active_laser_uid

                if not cmd.is_state_command():
                    ops.add(cmd)
                else:
                    raise ValueError(f"unexpected command {cmd}")
            reassemble_ctx.set_progress(i + 1)

        logger.debug("Optimization finished")
        context.set_message(_("Optimization complete"))
        context.set_progress(1.0)
        context.flush()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Optimize":
        """Creates an Optimize instance from a dictionary."""
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        # This transformer has no configurable parameters other than 'enabled'
        return cls(enabled=data.get("enabled", True))
