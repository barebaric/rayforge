import numpy as np
import math
from copy import copy
from ..models.ops import Ops, State, ArcToCommand
from .transformer import OpsTransformer


def split_long_segments(operations):
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
    for op in operations:
        if last_state.allow_rapid_change(op.state):
            segments[-1].append(op)
        else:
            # If rapid state change is not allowed, add
            # it to a new long segment.
            segments.append([op])
    return segments


def split_segments(commands):
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
            raise ValueError(f'unexpected Command {cmd}')

    if current_segment:
        segments.append(current_segment)
    return segments


def flip_segment(segment):
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
    for i in range(length-1, -1, -1):
        cmd = segment[i]
        prev_cmd = segment[(i+1) % length]
        new_cmd = copy(prev_cmd)
        new_cmd.end = cmd.end

        # Fix arc_to parameters
        if isinstance(new_cmd, ArcToCommand) and i > 0:
            # Get original arc (prev op in original segment)
            orig_cmd = segment[i+1]
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


def greedy_order_segments(segments):
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

    ordered = []
    current_seg = segments[0]
    ordered.append(current_seg)
    current_pos = np.array(current_seg[-1].end)
    remaining = segments[1:]
    while remaining:
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

    return ordered


def flip_segments(ordered):
    """
    Flip each segment if doing so lowers the sum of the incoming
    and outgoing travel.
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(ordered)):
            # Calculate cost of travel (=travel distance from last segment
            # +travel distance to next segment)
            prev_segment_end = ordered[i-1][-1].end
            segment = ordered[i]
            cost = math.dist(prev_segment_end, segment[0].end)
            if i < len(ordered)-1:
                cost += math.dist(segment[-1].end, ordered[i+1][0].end)

            # Flip and calculate the flipped cost.
            flipped = flip_segment(segment)
            flipped_cost = math.dist(prev_segment_end, flipped[0].end)
            if i < len(ordered)-1:
                flipped_cost += math.dist(flipped[-1].end,
                                          ordered[i+1][0].end)

            # Choose the shorter one.
            if flipped_cost < cost:
                ordered[i] = flipped
                improved = True

    return ordered


def two_opt(ordered, max_iter=1000):
    """
    2-opt: try reversing entire sub-sequences if that lowers the travel cost.
    """
    n = len(ordered)
    if n < 3:
        return ordered
    iter_count = 0
    improved = True
    while improved and iter_count < max_iter:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a_end = ordered[i][-1]
                b_start = ordered[i+1][0]
                e_end = ordered[j][-1]
                if j < n - 1:
                    f_start = ordered[j+1][0]
                    curr_cost = (
                        math.dist(a_end.end, b_start.end)
                        + math.dist(e_end.end, f_start.end)
                    )
                    new_cost = (
                        math.dist(a_end.end, e_end.end)
                        + math.dist(b_start.end, f_start.end)
                    )
                else:
                    curr_cost = math.dist(a_end.end, b_start.end)
                    new_cost = math.dist(a_end.end, e_end.end)
                if new_cost < curr_cost:
                    sub = ordered[i+1:j+1]
                    # Reverse order and flip each segment.
                    for n in range(len(sub)):
                        sub[n] = flip_segment(sub[n])
                    ordered[i+1:j+1] = sub[::-1]
                    improved = True
        iter_count += 1
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
    def run(self, ops: Ops):
        # 1. Preprocess such that each operation has a state.
        # This also causes all state commands to be dropped - we
        # need to re-add them later.
        ops.preload_state()
        commands = [c for c in ops if not c.is_state_command()]

        # 2. Split the operations into long segments where
        # the state stays more or less the same, i.e. no switching
        # of states that we should be careful with, such as toggling
        # air assist.
        long_segments = split_long_segments(commands)

        # 3. Split the long segments into small, re-orderable
        # segments.
        result = []
        for long_segment in long_segments:
            # 4. Reorder to minimize the distance.
            segments = split_segments(long_segment)
            segments = greedy_order_segments(segments)
            segments = flip_segments(segments)
            result += two_opt(segments, max_iter=1000)

        # 5. Reassemble the ops, reintroducing state change commands.
        ops.commands = []
        prev_state = State()
        for segment in result:
            if not segment:
                continue  # skip empty segments

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
                    raise ValueError(f'unexpected command {cmd}')
