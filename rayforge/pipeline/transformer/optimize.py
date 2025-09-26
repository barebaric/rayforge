import numpy as np
import math
import logging
from typing import Optional, List, Dict, Any, Tuple, cast
from ...core.workpiece import WorkPiece
from ...core.ops import (
    Ops,
    State,
    MovingCommand,
    Command,
    ScanLinePowerCommand,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)
from ...core.ops.flip import flip_segment
from ...core.ops.group import (
    group_by_state_continuity,
    group_by_path_continuity,
)
from .base import OpsTransformer, ExecutionPhase
from ...shared.tasker.context import BaseExecutionContext, ExecutionContext


logger = logging.getLogger(__name__)


def _dist_2d(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Helper for 2D distance calculation on n-dimensional points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def group_mixed_continuity(
    commands: List[MovingCommand],
) -> List[List[MovingCommand]]:
    """
    Splits a command list into continuous path segments. It correctly pairs
    a MoveTo command with a subsequent ScanLinePowerCommand to form a single
    optimizable raster segment. Vector paths are grouped as before.
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
            # This is a raster segment: [MoveTo, ScanLinePowerCommand]
            current_segment = [start_cmd, commands[i + 1]]
            segments.append(current_segment)
            i += 2  # Consume both commands
        else:
            # This is a standard vector segment. Consume all cutting commands.
            current_segment = [start_cmd]
            i += 1
            while i < len(commands) and commands[i].is_cutting_command():
                # Defensively handle mixed vector/raster types
                if isinstance(commands[i], ScanLinePowerCommand):
                    break
                current_segment.append(commands[i])
                i += 1
            segments.append(current_segment)

    return segments


def greedy_order_segments(
    context: BaseExecutionContext,
    segments: List[List[MovingCommand]],
) -> List[List[MovingCommand]]:
    """
    Greedy ordering using vectorized math.dist computations.
    Part of the path optimization algorithm.

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


def flip_segments(
    context: BaseExecutionContext, ordered: List[List[MovingCommand]]
) -> List[List[MovingCommand]]:
    """
    Iteratively flips individual segments if doing so reduces the travel
    distance to the next segment.
    """
    improved = True
    context.set_total(1)
    while improved:
        if context.is_cancelled():
            return ordered
        improved = False

        # Iterate through segments, comparing the cost of the current
        # orientation
        # with the cost if the segment were flipped.
        for i in range(1, len(ordered)):
            prev_segment = ordered[i - 1]
            current_segment = ordered[i]

            prev_end = prev_segment[-1].end
            curr_start = current_segment[0].end
            curr_end = current_segment[-1].end

            # Calculate cost with current orientation

            # Cost of travel from previous segment to this one
            current_cost = _dist_2d(prev_end, curr_start)

            # Calculate potential cost if flipped

            # Potential cost of travel from previous segment to the *end*
            # (flipped start)
            potential_cost = _dist_2d(prev_end, curr_end)

            # Factor in the cost to the next segment
            if i < len(ordered) - 1:
                next_start = ordered[i + 1][0].end

                # Add travel from current end to next start
                current_cost += _dist_2d(curr_end, next_start)

                # Add travel from flipped end (current start) to next start
                potential_cost += _dist_2d(curr_start, next_start)

            # If the potential cost is lower, commit to the flip.
            if potential_cost < current_cost:
                # flip_segment creates a new, correctly flipped segment.
                ordered[i] = flip_segment(current_segment)
                improved = True
    context.set_progress(1)
    return ordered


def farthest_insertion_order_segments(
    context: BaseExecutionContext, segments: List[List[MovingCommand]]
) -> List[List[MovingCommand]]:
    """
    Orders segments using the Farthest Insertion heuristic. This is a single-
    pass O(n^2) algorithm that is much faster than 2-opt for large n.
    """
    n = len(segments)
    if n < 2:
        return segments

    context.set_total(n)
    points = np.array([seg[0].end[:2] for seg in segments])

    # --- O(n^2) Farthest Insertion Implementation ---

    # 1. Initialization
    remaining_mask = np.ones(n, dtype=bool)
    start_idx = 0
    tour_indices = [start_idx]
    remaining_mask[start_idx] = False
    context.set_progress(1)
    dists = np.linalg.norm(points - points[start_idx], axis=1)

    while len(tour_indices) < n:
        if context.is_cancelled():
            break

        # 2. Select the farthest point.
        dists_of_remaining = np.where(remaining_mask, dists, -1.0)
        farthest_idx = int(np.argmax(dists_of_remaining))

        # 3. Find the best insertion position.
        min_cost_increase = float("inf")
        best_insert_pos = -1
        farthest_point = points[farthest_idx]
        for i in range(len(tour_indices)):
            p1_idx = tour_indices[i]
            p2_idx = tour_indices[(i + 1) % len(tour_indices)]
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            cost_increase = (
                np.linalg.norm(p1 - farthest_point)
                + np.linalg.norm(farthest_point - p2)
                - np.linalg.norm(p1 - p2)
            )
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_insert_pos = i + 1

        # 4. Insert the point and update state.
        tour_indices.insert(best_insert_pos, farthest_idx)
        remaining_mask[farthest_idx] = False
        context.set_progress(len(tour_indices))

        # 5. Update the minimum distances for all remaining points.
        new_dists_to_farthest = np.linalg.norm(points - farthest_point, axis=1)
        dists = np.minimum(dists, new_dists_to_farthest)

    ordered = [segments[i] for i in tour_indices]

    # 6. Final pass to ensure segments are correctly oriented (flipped).
    for i in range(n):
        current_segment = ordered[i]
        prev_end = ordered[i - 1][-1].end
        dist_as_is = _dist_2d(prev_end, current_segment[0].end)
        dist_flipped = _dist_2d(prev_end, current_segment[-1].end)
        if dist_flipped < dist_as_is:
            ordered[i] = flip_segment(current_segment)

    context.set_progress(n)
    return ordered


def two_opt(
    context: BaseExecutionContext,
    ordered: List[List[MovingCommand]],
    max_iter: int,
) -> List[List[MovingCommand]]:
    """
    2-opt: try reversing entire sub-sequences if that lowers the travel cost.
    Uses a simplified linear progress model based on the iteration count.
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


class Optimize(OpsTransformer):
    """
    Uses the 2-opt swap algorithm to address the Traveline Salesman Problem
    to minimize travel moves in the commands.

    This is made harder by the fact that some commands cannot be
    reordered. For example, if the Ops contains multiple commands
    to toggle air-assist, we cannot reorder the operations without
    ensuring that air-assist remains on for the sections that need it.
    Ops optimization may lead to a situation where the number of
    air assist toggles is multiplied, which could be detrimental
    to the health of the air pump.

    To avoid these problems, we implement the following process:

    1. Preprocess the command list, duplicating the intended
       state (e.g. cutting, power, ...) and attaching it to each
       command. Here we also drop all state commands.

    2. Split the command list into non-reorderable segments. Segment in
       this step means an "as long as possible" sequence that may still
       include sub-segments, as long as those sub-segments are
       reorderable.

    3. Split the long segments into short, re-orderable sub sequences.

    4. Re-order the sub sequences to minimize travel distance.

    5. Re-assemble the Ops object.
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
        return _("Minimizes travel distance by reordering segments")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[BaseExecutionContext] = None,
    ) -> None:
        if context is None:
            context = ExecutionContext()

        # Threshold for switching from 2-opt to a faster heuristic
        TWO_OPT_THRESHOLD = 1000

        # Step 1: Preprocessing
        context.set_message(_("Preprocessing for optimization..."))
        ops.preload_state()
        if context.is_cancelled():
            return

        # Expand ScanLinePowerCommands into smaller, optimizable segments.
        # This makes raster optimization possible by breaking up full-width
        # scans.
        RASTER_MIN_POWER = 0  # power level (0-255) to be considered "off"

        expanded_commands: List[Command] = []
        last_end_pos = (0.0, 0.0, 0.0)

        for cmd in ops.commands:
            if isinstance(cmd, ScanLinePowerCommand):
                start_point_for_scan = last_end_pos
                sub_commands = cmd.split_by_power(
                    start_point_for_scan, RASTER_MIN_POWER
                )

                if sub_commands:
                    # The sub_commands contain new MoveTo commands. This
                    # makes the MoveTo that preceded the original ScanLine
                    # redundant. Remove it.
                    if (
                        expanded_commands
                        and expanded_commands[-1].is_travel_command()
                        and expanded_commands[-1].end == start_point_for_scan
                    ):
                        expanded_commands.pop()

                if cmd.state:
                    for sub_cmd in sub_commands:
                        sub_cmd.state = cmd.state
                expanded_commands.extend(sub_commands)
            else:
                expanded_commands.append(cmd)

            if isinstance(cmd, MovingCommand) and cmd.end is not None:
                last_end_pos = cmd.end

        # The rest of the optimization process uses this expanded command list
        commands = [c for c in expanded_commands if not c.is_state_command()]
        logger.debug(
            f"Original command count {len(ops.commands)},"
            f" expanded for optimization to {len(commands)}"
        )

        # Step 2: Splitting long segments
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

        # Pre-calculate the number of sub-segments in each long segment to
        # create a weighted progress bar that reflects the actual workload.
        is_raster_section = False
        segment_workloads = []
        for long_segment in long_segments:
            workload = 0
            if long_segment and not long_segment[0].is_marker_command():
                # This mirrors the logic in the main loop to determine segment
                # type
                marker = next(
                    (c for c in long_segment if c.is_marker_command()), None
                )
                if isinstance(marker, OpsSectionStartCommand):
                    is_raster_section = (
                        marker.section_type == SectionType.RASTER_FILL
                    )
                # Now, calculate the number of sub-segments
                if is_raster_section:
                    sub_segments = group_mixed_continuity(
                        cast(List[MovingCommand], long_segment)
                    )
                else:
                    sub_segments = group_by_path_continuity(long_segment)
                workload = len(sub_segments)
            segment_workloads.append(max(1, workload))  # Min 1 unit of work

        total_workload = sum(segment_workloads)
        cumulative_workload = 0.0

        result = []
        is_raster_section = False  # Reset for the main loop
        for i, long_segment in enumerate(long_segments):
            if context.is_cancelled():
                return

            # If the segment is a marker, just pass it through without
            # optimizing
            current_workload = segment_workloads[i]
            progress_range = (
                current_workload / total_workload if total_workload > 0 else 0
            )
            base_progress = (
                cumulative_workload / total_workload
                if total_workload > 0
                else 0
            )

            if long_segment and long_segment[0].is_marker_command():
                marker = long_segment[0]
                if isinstance(marker, OpsSectionStartCommand):
                    if marker.section_type == SectionType.RASTER_FILL:
                        is_raster_section = True
                elif isinstance(marker, OpsSectionEndCommand):
                    if marker.section_type == SectionType.RASTER_FILL:
                        is_raster_section = False
                result.append(long_segment)
                cumulative_workload += current_workload
                optimize_ctx.set_progress(
                    cumulative_workload / total_workload
                    if total_workload > 0
                    else 1.0
                )
                continue

            context.set_message(
                _("Optimizing segment {i}/{total}...").format(
                    i=i + 1, total=len(long_segments)
                )
            )

            segment_ctx = optimize_ctx.sub_context(
                base_progress=base_progress, progress_range=progress_range
            )

            # Define weights for the internal stages of optimizing one
            # segment.
            split_sub_weight = 0.05
            greedy_weight = 0.4
            flip_weight = 0.2
            refinement_weight = 0.35

            # Step 3: Split long segments into short segments
            context.set_message(_("Finding reorderable paths..."))

            # Use the appropriate grouping strategy based on section type
            if is_raster_section:
                segments = group_mixed_continuity(
                    cast(List[MovingCommand], long_segment)
                )
            else:
                segments = group_by_path_continuity(long_segment)

            # Mark this small stage as complete.
            segment_ctx.set_progress(split_sub_weight)
            if not segments:
                cumulative_workload += current_workload
                continue

            current_base = split_sub_weight

            # Step 4: Re-order segments
            # First, order them using a greedy algorithm
            context.set_message(_("Ordering paths..."))
            greedy_ctx = segment_ctx.sub_context(
                base_progress=current_base, progress_range=greedy_weight
            )
            segments = greedy_order_segments(greedy_ctx, segments)
            current_base += greedy_weight
            segment_ctx.set_progress(current_base)

            # Second, flip segments to shorten distance (fast, no detailed
            # progress needed)
            context.set_message(_("Flipping segments for improvement..."))
            flip_ctx = segment_ctx.sub_context(
                base_progress=current_base, progress_range=flip_weight
            )
            segments = flip_segments(flip_ctx, segments)
            current_base += flip_weight
            segment_ctx.set_progress(current_base)

            # Apply 2-opt algorithm
            refinement_ctx = segment_ctx.sub_context(
                base_progress=current_base, progress_range=refinement_weight
            )
            if len(segments) >= TWO_OPT_THRESHOLD:
                logger.info(
                    f"Segment count ({len(segments)}) is high, "
                    f"falling back to Farthest Insertion instead of 2-opt."
                )
                refinement_ctx.set_message(
                    _("Using Farthest Insertion for large path...")
                )
                segments = farthest_insertion_order_segments(
                    refinement_ctx, segments
                )
            else:
                refinement_ctx.set_message(_("Applying 2-opt refinement..."))
                segments = two_opt(refinement_ctx, segments, 10)

            result.append(segments)
            cumulative_workload += current_workload

        # Ensure the optimization part reports full completion.
        optimize_ctx.set_progress(1.0)

        # Step 5: Re-assemble the Ops object.
        context.set_message(_("Reassembling optimized paths..."))
        reassemble_ctx = context.sub_context(
            base_progress=optimize_weight, progress_range=reassemble_weight
        )

        flat_result_segments = []
        for item in result:
            if item and isinstance(item[0], list):
                flat_result_segments.extend(item)
            else:
                flat_result_segments.append(item)
        reassemble_ctx.set_total(len(flat_result_segments))
        ops.commands = []
        prev_state = State()
        for i, segment in enumerate(flat_result_segments):
            if not segment:
                continue

            if segment[0].is_marker_command():
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
