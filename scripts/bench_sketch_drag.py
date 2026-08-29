"""Benchmark for interactive drag performance in the sketcher.

Builds a sketch of N independent dimensioned rectangles plus one
freely movable target rectangle, then times the work performed per
mouse-move during a simulated edge drag, once with a global solve
and once with a component-scoped solve (point_scope) as used by
the select tool while dragging:

- the pre-solve magnetic snap query
- Sketch.solve() with strong drag constraints and weak hold
  constraints for the scoped/global point set
- the post-solve snap feedback query

Run from the repository root:

    python3 scripts/bench_sketch_drag.py [--islands 10 25 50] \
        [--moves 20] [--profile]
"""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ADDON_DIR = (
    _REPO_ROOT / "rayforge" / "builtin_addons" / "rayforge-addon-sketcher"
)
sys.path.insert(0, str(_ADDON_DIR))

from sketcher.core.components import (  # noqa: E402
    compute_constraint_components,
)
from sketcher.core.constraints import DragConstraint  # noqa: E402
from sketcher.core.sketch import Sketch  # noqa: E402
from sketcher.core.snap import SnapEngine  # noqa: E402
from sketcher.core.snap.producers import (  # noqa: E402
    CentersProducer,
    EntityPointsProducer,
    EquidistantLinesProducer,
    IntersectionsProducer,
    MidpointsProducer,
    OnEntityProducer,
)
from sketcher.core.snap.types import DragContext  # noqa: E402

STRONG_DRAG_WEIGHT = 1.0
HOLD_WEIGHT = 0.01
DRAG_STEP = 1.0


def add_constrained_rect(sketch, x, y, w, h, anchored):
    """Adds a rectangle with horizontal/vertical constraints.

    If anchored, the rectangle is additionally fully constrained via
    two dimensions and a fixed corner. Otherwise one edge can move,
    which makes it draggable.
    """
    p1 = sketch.add_point(x, y)
    p2 = sketch.add_point(x + w, y)
    p3 = sketch.add_point(x + w, y + h)
    p4 = sketch.add_point(x, y + h)
    sketch.add_line(p1, p2)
    sketch.add_line(p2, p3)
    sketch.add_line(p3, p4)
    sketch.add_line(p4, p1)
    sketch.constrain_horizontal(p1, p2)
    sketch.constrain_horizontal(p4, p3)
    sketch.constrain_vertical(p1, p4)
    sketch.constrain_vertical(p2, p3)
    if anchored:
        sketch.constrain_distance(p1, p2, w)
        sketch.constrain_distance(p1, p4, h)
        sketch.registry.get_point(p1).fixed = True
    return [p1, p2, p3, p4]


def build_sketch(num_islands):
    sketch = Sketch("bench")
    for i in range(num_islands):
        col = i % 10
        row = i // 10
        add_constrained_rect(
            sketch, col * 40.0, row * 40.0, 20.0, 10.0, anchored=True
        )
    target_points = add_constrained_rect(
        sketch,
        (num_islands % 10) * 40.0 + 5.0,
        (num_islands // 10) * 40.0 + 20.0,
        20.0,
        10.0,
        anchored=False,
    )
    return sketch, target_points


def compute_scope(sketch, target_points):
    """Mirrors SelectTool._compute_drag_scope."""
    scope = set(target_points)
    for component in compute_constraint_components(
        sketch.registry, sketch.constraints
    ):
        if not component.isdisjoint(scope):
            scope |= component
    return scope


def build_drag_constraints(sketch, target_points, dx, scope):
    """Mirrors _handle_entity_drag: strong constraints on the dragged
    points, weak hold constraints on every other solved point."""
    constraints = []
    for pid in target_points:
        p = sketch.registry.get_point(pid)
        if p.fixed:
            continue
        constraints.append(
            DragConstraint(pid, p.x + dx, p.y, weight=STRONG_DRAG_WEIGHT)
        )
    dragged = set(target_points)
    for p in sketch.registry.points:
        if p.fixed or p.id in dragged:
            continue
        if scope is not None and p.id not in scope:
            continue
        constraints.append(DragConstraint(p.id, p.x, p.y, weight=HOLD_WEIGHT))
    return constraints


def create_snap_engine():
    engine = SnapEngine()
    engine.register_producer(EntityPointsProducer())
    engine.register_producer(OnEntityProducer())
    engine.register_producer(MidpointsProducer())
    engine.register_producer(IntersectionsProducer())
    engine.register_producer(EquidistantLinesProducer())
    engine.register_producer(CentersProducer())
    return engine


def run_moves(
    sketch,
    target_points,
    snap_engine,
    num_moves,
    moves_times,
    snap_times,
    scope,
):
    drag_context = DragContext(
        dragged_point_ids=set(target_points),
        dragged_entity_ids=set(),
    )
    origin = sketch.registry.get_point(target_points[0])
    for step in range(1, num_moves + 1):
        dx = DRAG_STEP if step % 2 else -DRAG_STEP
        drag_constraints = build_drag_constraints(
            sketch, target_points, dx, scope
        )
        query_pos = (origin.x + dx, origin.y)

        start = time.perf_counter()
        snap_engine.query(sketch.registry, query_pos, drag_context)
        snap_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        sketch.solve(
            extra_constraints=drag_constraints,
            update_constraint_status=False,
            point_scope=scope,
        )
        moves_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        snap_engine.query(sketch.registry, query_pos, drag_context)
        snap_times.append(time.perf_counter() - start)


def measure(num_islands, num_moves, scoped):
    sketch, target_points = build_sketch(num_islands)
    scope = compute_scope(sketch, target_points) if scoped else None
    snap_engine = create_snap_engine()
    moves_times = []
    snap_times = []
    run_moves(
        sketch,
        target_points,
        snap_engine,
        num_moves,
        moves_times,
        snap_times,
        scope,
    )
    solve_ms = 1000.0 * sum(moves_times) / len(moves_times)
    snap_ms = 1000.0 * sum(snap_times) / len(snap_times)
    return solve_ms, snap_ms


def profile_solve(num_islands, num_moves, scoped):
    sketch, target_points = build_sketch(num_islands)
    scope = compute_scope(sketch, target_points) if scoped else None
    profiler = cProfile.Profile()
    profiler.enable()
    for step in range(1, num_moves + 1):
        dx = DRAG_STEP if step % 2 else -DRAG_STEP
        drag_constraints = build_drag_constraints(
            sketch, target_points, dx, scope
        )
        sketch.solve(
            extra_constraints=drag_constraints,
            update_constraint_status=False,
            point_scope=scope,
        )
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--islands",
        type=int,
        nargs="+",
        default=[1, 10, 25, 50, 100],
        help="Island counts to measure",
    )
    parser.add_argument(
        "--moves",
        type=int,
        default=20,
        help="Simulated mouse-move steps per configuration",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="cProfile the scoped solve loop for the largest island count",
    )
    args = parser.parse_args()

    print(
        f"{'islands':>8} {'points':>7} {'constr':>7} "
        f"{'global solve':>13} {'scoped solve':>13} {'snap/query':>11}"
    )
    for num_islands in args.islands:
        global_ms, _ = measure(num_islands, args.moves, scoped=False)
        scoped_ms, snap_ms = measure(num_islands, args.moves, scoped=True)
        sketch, _ = build_sketch(num_islands)
        print(
            f"{num_islands:>8} {len(sketch.registry.points):>7} "
            f"{len(sketch.constraints):>7} {global_ms:>11.2f}ms "
            f"{scoped_ms:>11.2f}ms {snap_ms:>9.2f}ms"
        )

    if args.profile:
        profile_solve(max(args.islands), args.moves, scoped=True)


if __name__ == "__main__":
    main()
