import numpy as np
import math
import logging
from copy import copy
from typing import Optional, List, cast
from ..models.ops import Ops, State, ArcToCommand, Command
from .transformer import OpsTransformer
from ..task import ExecutionContext


logger = logging.getLogger(__name__)


def split_long_segments(operations: List[Command]) -> List[List[Command]]:
    """
    Split a list of operations such that segments where air assist
    is enabled are separated from segments where it is not. We
    need this because these segments must remain in order,
    so we need to separate them and run the path optimizer on
    each segment individually.

    The result is a list of Command lists.
    """
    if len(operations) <= 1:
        return [operations]

    segments = [[operations[0]]]
    last_state = operations[0].state
    for op in operations[1:]:
        if last_state.allow_rapid_change(op.state):
            segments[-1].append(op)
        else:
            # If rapid state change is not allowed, add
            # it to a new long segment.
            segments.append([op])
            last_state = op.state
    return segments


def split_segments(commands: List[Command]) -> List[List[Command]]:
    """
    Split a list of commands into segments. We use it to prepare
    for reordering the segments for travel distance minimization.

    Returns a list of segments. In other words, a list of list[Command].
    """
    segments = []
    current_segment = []
    for cmd in commands:
        if cmd.is_travel_command():
            if current_segment:
                segments.append(current_segment)
            current_segment = [cmd]
        elif cmd.is_cutting_command():
            current_segment.append(cmd)
        else:
            raise ValueError(f"unexpected Command {cmd}")

    if current_segment:
        segments.append(current_segment)
    return segments


def flip_segment(segment: List[Command]) -> List[Command]:
    """
    The states attached to each point descibe the intended
    machine state while traveling TO the point.

    Example:
      state:     A            B            C           D
      points:   -> move_to 1 -> line_to 2 -> arc_to 3 -> line_to 4

    After flipping this sequence, the state is in the wrong position:

      state:     D            C           B            A
      points:   -> line_to 4 -> arc_to 3 -> line_to 2 -> move_to 1

    Note that for example the edge between point 3 and 2 no longer has
    state C, it is B instead. 4 -> 3 should be D, but is C.
    So we have to shift the state and the command to the next point.
    Correct:

      state:     A            D            C           B
      points:   -> move_to 4 -> line_to 3 -> arc_to 2 -> line_to 1
    """
    length = len(segment)
    if length <= 1:
        return segment

    new_segment = []
    for i in range(length - 1, -1, -1):
        cmd = segment[i]
        prev_cmd = segment[(i + 1) % length]
        new_cmd = copy(prev_cmd)
        new_cmd.end = cmd.end

        # Fix arc_to parameters
        if isinstance(new_cmd, ArcToCommand) and i > 0:
            # Get original arc (prev op in original segment)
            orig_cmd = cast(ArcToCommand, segment[i + 1])
            x_end, y_end = orig_cmd.end
            i_orig, j_orig = orig_cmd.center_offset

            # Calculate center and new offsets
            x_start, y_start = new_cmd.end
            center_x = x_start + i_orig
            center_y = y_start + j_orig
            new_i = center_x - x_end
            new_j = center_y - y_end

            # Update arc parameters
            new_cmd.end = x_start, y_start
            new_cmd.center_offset = new_i, new_j
            new_cmd.clockwise = not orig_cmd.clockwise

        new_segment.append(new_cmd)

    return new_segment


def greedy_order_segments(
    context: ExecutionContext,
    segments: List[List[Command]],
) -> List[List[Command]]:
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

    context.set_total(len(segments))
    ordered: List[List[Command]] = []
    current_seg = segments[0]
    ordered.append(current_seg)
    current_pos = np.array(current_seg[-1].end)
    remaining = segments[1:]
    context.set_progress(1)

    while remaining:
        if context.is_cancelled():
            return ordered

        # Find the index of the best next path to take, i.e. the
        # Command that adds the smalles amount of travel distance.
        starts = np.array([seg[0].end for seg in remaining])
        ends = np.array([seg[-1].end for seg in remaining])
        d_starts = np.linalg.norm(starts - current_pos, axis=1)
        d_ends = np.linalg.norm(ends - current_pos, axis=1)
        candidate_dists = np.minimum(d_starts, d_ends)
        best_idx = int(np.argmin(candidate_dists))
        best_seg = remaining.pop(best_idx)

        # Flip candidate if its end is closer.
        if d_ends[best_idx] < d_starts[best_idx]:
            best_seg = flip_segment(best_seg)

        start_cmd = best_seg[0]
        if not start_cmd.is_travel_command():
            end_cmd = best_seg[-1]
            best_seg[0], best_seg[-1] = best_seg[-1], best_seg[0]
            start_cmd.end, end_cmd.end = end_cmd.end, start_cmd.end

        ordered.append(best_seg)
        current_pos = np.array(best_seg[-1].end)
        context.set_progress(len(ordered))

    return ordered


def flip_segments(
    context: ExecutionContext, ordered: List[List[Command]]
) -> List[List[Command]]:
    improved = True
    while improved:
        if context.is_cancelled():
            return ordered
        improved = False
        for i in range(1, len(ordered)):
            # Calculate cost of travel (=travel distance from last segment
            # +travel distance to next segment)
            prev_segment_end = ordered[i - 1][-1].end
            segment = ordered[i]
            cost = math.dist(prev_segment_end, segment[0].end)
            if i < len(ordered) - 1:
                cost += math.dist(segment[-1].end, ordered[i + 1][0].end)

            # Flip and calculate the flipped cost.
            flipped = flip_segment(segment)
            flipped_cost = math.dist(prev_segment_end, flipped[0].end)
            if i < len(ordered) - 1:
                flipped_cost += math.dist(
                    flipped[-1].end, ordered[i + 1][0].end
                )

            # Choose the shorter one.
            if flipped_cost < cost:
                ordered[i] = flipped
                improved = True

    return ordered


def two_opt(
    context: ExecutionContext,
    ordered: List[List[Command]],
    max_iter: int,
) -> List[List[Command]]:
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
                a_end = ordered[i][-1]
                b_start = ordered[i + 1][0]
                e_end = ordered[j][-1]
                if j < n - 1:
                    f_start = ordered[j + 1][0]
                    curr_cost = math.dist(a_end.end, b_start.end) + math.dist(
                        e_end.end, f_start.end
                    )
                    new_cost = math.dist(a_end.end, e_end.end) + math.dist(
                        b_start.end, f_start.end
                    )
                else:
                    curr_cost = math.dist(a_end.end, b_start.end)
                    new_cost = math.dist(a_end.end, e_end.end)
                if new_cost < curr_cost:
                    sub = ordered[i + 1:j + 1]
                    # Reverse order and flip each segment.
                    for n in range(len(sub)):
                        sub[n] = flip_segment(sub[n])
                    ordered[i + 1:j + 1] = sub[::-1]
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

    def run(
        self, ops: Ops, context: Optional[ExecutionContext] = None
    ) -> None:
        if context is None:
            context = ExecutionContext()

        # Step 1: Preprocessing
        context.set_message(_("Preprocessing for optimization..."))
        ops.preload_state()
        if context.is_cancelled():
            return
        commands = [c for c in ops if not c.is_state_command()]
        logger.debug(f"Command count {len(commands)}")

        # Step 2: Splitting long segments
        long_segments = split_long_segments(commands)
        if context.is_cancelled():
            return

        result = []
        total_long_segments = len(long_segments) if long_segments else 1
        optimize_ctx = context.sub_context(
            base_progress=0,
            progress_range=0.8,
            total=total_long_segments,
        )
        for i, long_segment in enumerate(long_segments):
            if context.is_cancelled():
                return

            context.set_message(
                _("Optimizing segment {i}/{total}...").format(
                    i=i + 1, total=total_long_segments
                )
            )

            # This sub-context manages the 4 optimization steps for one segment
            segment_ctx = optimize_ctx.sub_context(
                base_progress=i, progress_range=1, total=4
            )

            # Step 3: Split long segments into short segments
            segments = split_segments(long_segment)
            segment_ctx.set_progress(1)
            if context.is_cancelled():
                return

            # Step 4: Re-order segments
            # First, order them using a greedy algorithm
            greedy_ctx = segment_ctx.sub_context(
                base_progress=1, progress_range=1
            )
            segments = greedy_order_segments(greedy_ctx, segments)
            segment_ctx.set_progress(2)
            if context.is_cancelled():
                return

            # Second, flip segments to shorten distance (fast, no detailed
            # progress needed)
            flip_ctx = segment_ctx.sub_context(
                base_progress=2, progress_range=1
            )
            segments = flip_segments(flip_ctx, segments)
            segment_ctx.set_progress(3)
            if context.is_cancelled():
                return

            # Apply 2-opt algorithm (has detailed progress)
            two_opt_ctx = segment_ctx.sub_context(
                base_progress=3, progress_range=1
            )
            result += two_opt(two_opt_ctx, segments, 1000)
            segment_ctx.set_progress(4)
            if context.is_cancelled():
                return

        # Step 5: Re-assemble the Ops object.
        context.set_message(_("Reassembling optimized paths..."))
        total_reassemble_segments = len(result) if result else 1
        reassemble_ctx = context.sub_context(
            base_progress=0.8,
            progress_range=0.2,
            total=total_reassemble_segments,
        )
        ops.commands = []
        prev_state = State()
        for i, segment in enumerate(result):
            if not segment:
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
