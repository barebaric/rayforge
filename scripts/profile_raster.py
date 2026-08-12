#!/usr/bin/env python3
"""Raster-engraving performance & memory harness.

Drives the real production functions for the three cost layers of the
raster pipeline and reports, per layer:

  * wall time
  * native-heap peak delta  (glibc ``mallinfo2`` — captures Rust + numpy)
  * Python-heap peak delta  (``tracemalloc``)
  * RSS peak delta          (``/proc/self/status`` sampled at ~5 ms)

Layers measured
---------------
A. Python preprocess   pyvips load -> render to Cairo surface
                        -> ``preprocess_raster_image`` -> numpy arrays
B. PyO3 marshalling    ``WholeImageSource(array)`` + alpha ``.tobytes()``
                        (this is where ``extract_flat_u8`` runs)
C. Rust assembly       ``RasterSpec`` + ``Assembler`` via ``execute_stages``
                        (scan-line generation + ``Ops`` emission)

The dimension/interval math mirrors
``EngraveStep._build_raster_part`` / ``build_compute_payload`` so the
numbers match the live app for a given laser spot size and workpiece size.

Usage
-----
    pixi run python scripts/profile_raster.py media/test-images/wolf.png
    pixi run python scripts/profile_raster.py IMG --spot-mm 0.1 --mode power_modulated
    pixi run python scripts/profile_raster.py IMG --sweep 0.05,0.1,0.2,0.4
    pixi run python scripts/profile_raster.py IMG --verbose   # top allocators

Profiling the real app (wolf.ryp) with memray --native
------------------------------------------------------
This harness measures only the compute + aggregate stages headlessly. To
attribute the *full app* memory (machine-transform, the G-code encoder,
render bitmaps, GUI), profile the live app loading a ``.ryp`` under a
virtual framebuffer with ``memray --native`` so Rust/raygeo allocations
are captured.

1. Make a tiny launcher so memray gets clean argv and you can target the
   process by PID (avoids ``pkill`` self-match — see caveats)::

       # /tmp/opencode/launch_rayforge.py
       import os, sys
       with open("/tmp/opencode/app.pid", "w") as f:
           f.write(str(os.getpid()))
       from rayforge.app import main
       sys.exit(main())

2. Run the app under xvfb + memray --native, time-bounded (the build is
   heavy under instrumentation; ~3-4 min budget)::

       timeout --signal=TERM 240 \\
         xvfb-run -a -s "-screen 0 1920x1080x24" \\
         pixi run python -m memray run --native --follow-fork \\
           -o /tmp/opencode/wolf_native.bin \\
           /tmp/opencode/launch_rayforge.py wolf.ryp

3. Analyze (``memray stats`` on a multi-GB bin is SLOW — allow minutes;
   invoke the env python directly to avoid a pixi re-solve)::

       .pixi/envs/default/bin/python -m memray stats  /tmp/opencode/wolf_native.bin
       .pixi/envs/default/bin/python -m memray flamegraph /tmp/opencode/wolf_native.bin

Caveats / lessons learned (this session)
----------------------------------------
* **``pixi run`` reinstalls raygeo from PyPI.** Any plain ``pixi run``
  re-solves the env and OVERWRITES a locally-built raygeo, silently
  discarding your Rust changes. To test local raygeo, add an override to
  ``pixi.toml`` and rebuild with ``pixi reinstall raygeo``::

      [pypi-options.dependency-overrides]
      raygeo = { path = "external/raygeo", editable = true }

  ``scripts/rebuild-raygeo.sh`` appends its OWN override and CONFLICTS with
  a manually-added one; use ``pixi reinstall raygeo`` instead. Remove the
  override before committing.

* **``scripts/pixi-raygeo.sh`` does not restore pixi.toml on SIGTERM.** Its
  cleanup trap is EXIT-only, so ``timeout``-killing it leaves a stale
  override marker in pixi.toml (which then blocks the wrapper). For
  long-running, timeout-killed app runs, prefer the manual override above.

* **rayforge is single-instance.** If an instance is already running, a
  second launch exits immediately and memray captures only imports (tiny
  bin, ~160 MB). Close the running instance first.

* **Never ``pkill -f rayforge``** — it matches the shell running the pkill
  and the command never finishes. Kill by PID via the launcher's
  ``/tmp/opencode/app.pid``.

* The app spams harmless serial-port errors (``/dev/ttyUSB0``) under xvfb.
  Ignore them.


Baseline attribution (wolf.ryp, ~7.5 GB peak)
---------------------------------------------
The peak is glibc retention of ~60 GB of allocation churn, not live data.
Top sources (memray --native, stock raygeo):
* native raygeo Ops copies (~38 GB churn) — addressed by ``Arc<Vec<OpNode>>``
  copy-on-write + boxed ``OpNode::state``.
* G-code encoder ``Ops::to_gcode`` (~12 GB / 147M allocs) — per-command
  ``format!()`` strings + ``op_to_machine_code`` / ``machine_code_to_op``
  ``HashMap``s. The maps are required by the simulator/G-code preview; they
  use ``HashMap`` for dense integer keys and could be compressed (dense
  ``Vec`` / run-length) for several-fold reduction.
* ``kinematic_mapping.apply_to_job_ops`` (~1.3 GB) — ``transform_layers``
  copies the whole job per layer even for flat (non-rotary) jobs; early-out
  when the machine has no ``rotary_modules``.
* ``surface_to_grayscale`` float32 alpha (~1.3 GB) — INTENTIONAL: cairo
  needs float32; do not "compress" it.
"""

import argparse
import ctypes
import gc
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pyvips
from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    ComputePayload,
    MachineParams,
)
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.raster import RasterSpec
from raygeo.ops.part import Part
from raygeo.ops.part.image_source import WholeImageSource
from raygeo.pipeline.execute import Pipeline
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

from rayforge.image.util.vips import (
    normalize_to_rgba,
    vips_rgba_to_cairo_surface,
)
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    preprocess_raster_image,
)

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _aggregate_node(key: str, source_keys: list[str]) -> NodeRequest:
    return NodeRequest(
        key=key,
        generation_id=1,
        stage=StageSpec.Aggregate(
            spec=AggregateSpec(
                wrap_start=[],
                groups=[
                    AggregateGroup(
                        start_markers=[],
                        inputs=[
                            AggregateInput(
                                source_key=sk,
                                placement_matrix=IDENTITY,
                                uid="",
                                target_dimensions=(0.0, 0.0),
                            )
                            for sk in source_keys
                        ],
                        end_markers=[],
                    )
                ],
                wrap_end=[],
                machine=MachineParams(),
                transformers=[],
            )
        ),
    )


MAX_RASTER_RENDER_PIXELS = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Native-heap measurement (glibc malloc)
# ---------------------------------------------------------------------------


def _make_mallinfo_struct(field_type):
    class _Mallinfo(ctypes.Structure):
        _fields_ = [
            ("arena", field_type),
            ("ordblks", field_type),
            ("smblks", field_type),
            ("hblks", field_type),
            ("hblkhd", field_type),
            ("usmblks", field_type),
            ("fsmblks", field_type),
            ("uordblks", field_type),
            ("fordblks", field_type),
            ("keepcost", field_type),
        ]

    return _Mallinfo


class Mallinfo:
    """Read glibc's in-use heap bytes (``uordblks``).

    ``mallinfo2`` (glibc >= 2.33) returns 64-bit fields; we fall back to
    the deprecated ``mallinfo`` (``int``) on older systems. Rust and
    numpy both allocate through the system allocator, so this captures
    the transient spikes (e.g. the ``.tolist()`` path) that RSS is too
    lazy to reflect.
    """

    def __init__(self):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        try:
            libc.mallinfo2.restype = _make_mallinfo_struct(ctypes.c_size_t)
            self._fn = libc.mallinfo2
        except AttributeError:
            libc.mallinfo.restype = _make_mallinfo_struct(ctypes.c_int)
            self._fn = libc.mallinfo

    def in_use_bytes(self) -> int:
        """Total bytes currently allocated via malloc (uordblks)."""
        return int(self._fn().uordblks)


def _probe_mallinfo() -> Mallinfo | None:
    try:
        mi = Mallinfo()
        mi.in_use_bytes()
        return mi
    except (OSError, AttributeError):
        return None


def read_rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class PhaseRec:
    name: str
    elapsed: float = 0.0
    base_arena: int = 0
    base_rss: int = 0
    base_traced: int = 0
    peak_arena_delta: int = 0
    peak_rss_delta: int = 0
    peak_traced_delta: int = 0
    end_traced_delta: int = 0


@dataclass
class Sampler:
    interval: float = 0.005
    mallinfo: Mallinfo | None = field(default_factory=_probe_mallinfo)
    phases: dict[str, PhaseRec] = field(default_factory=dict)
    _cur: PhaseRec | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            arena = (
                self.mallinfo.in_use_bytes()
                if self.mallinfo is not None
                else 0
            )
            rss = read_rss_kb() * 1024
            rec = self._cur
            if rec is not None:
                rec.peak_arena_delta = max(
                    rec.peak_arena_delta, arena - rec.base_arena
                )
                rec.peak_rss_delta = max(
                    rec.peak_rss_delta, rss - rec.base_rss
                )
            self._stop.wait(self.interval)

    @contextmanager
    def phase(self, name: str):
        gc.collect()
        tracemalloc.reset_peak()
        base_arena = self.mallinfo.in_use_bytes() if self.mallinfo else 0
        rec = PhaseRec(
            name=name,
            base_arena=base_arena,
            base_rss=read_rss_kb() * 1024,
            base_traced=tracemalloc.get_traced_memory()[0],
        )
        self.phases[name] = rec
        self._cur = rec
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            t1 = time.perf_counter()
            self._cur = None
            rec.elapsed = t1 - t0
            cur_tr, peak_tr = tracemalloc.get_traced_memory()
            rec.peak_traced_delta = max(0, peak_tr - rec.base_traced)
            rec.end_traced_delta = max(0, cur_tr - rec.base_traced)
            gc.collect()


# ---------------------------------------------------------------------------
# Dimension math (mirrors EngraveStep._build_raster_part)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterGeom:
    target_w: int
    target_h: int
    px_per_mm_x: float
    px_per_mm_y: float
    line_interval_mm: float
    sample_interval_mm: float


def compute_geom(spot_mm: float, size_mm: tuple[float, float]) -> RasterGeom:
    spot_x = spot_y = spot_mm
    sample_interval = spot_x / 2.0
    px_per_mm_x = 1.0 / sample_interval
    px_per_mm_y = 1.0 / spot_y
    target_w = max(1, int(size_mm[0] * px_per_mm_x))
    target_h = max(1, int(size_mm[1] * px_per_mm_y))
    n = target_w * target_h
    if n > MAX_RASTER_RENDER_PIXELS:
        scale = (MAX_RASTER_RENDER_PIXELS / n) ** 0.5
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))
    px_per_mm_x = target_w / size_mm[0]
    px_per_mm_y = target_h / size_mm[1]
    return RasterGeom(
        target_w=target_w,
        target_h=target_h,
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
        line_interval_mm=spot_y,
        sample_interval_mm=sample_interval,
    )


# ---------------------------------------------------------------------------
# The three layers
# ---------------------------------------------------------------------------

DEPTH_BY_RAYGEO_NAME = {
    "power_modulated": DepthMode.POWER_MODULATION,
    "mask_scan": DepthMode.CONSTANT_POWER,
    "dither": DepthMode.DITHER,
    "multi_pass": DepthMode.MULTI_PASS,
}


def load_source_image(path: str) -> pyvips.Image:
    return pyvips.Image.pngload(path, access=pyvips.Access.RANDOM)


def layer_a_preprocess(
    src: pyvips.Image,
    geom: RasterGeom,
    mode: str,
) -> tuple[
    object, tuple[np.ndarray | None, np.ndarray | None], tuple[float, float]
]:
    """Render to a Cairo surface at the target resolution and run the
    real ``preprocess_raster_image``. The Part is built in layer B so
    marshalling is isolated from preprocessing."""
    rendered = src.thumbnail_image(
        geom.target_w, height=geom.target_h, size="force"
    )
    norm = normalize_to_rgba(rendered)
    surface = vips_rgba_to_cairo_surface(norm)
    depth = DEPTH_BY_RAYGEO_NAME[mode]
    image, alpha = preprocess_raster_image(
        surface,
        mode=depth,
        invert=False,
        auto_levels=True,
        laser_spot_x_mm=0.1,
        pixels_per_mm_x=geom.px_per_mm_x,
    )
    surface.flush()
    size_mm = (
        geom.target_w / geom.px_per_mm_x,
        geom.target_h / geom.px_per_mm_y,
    )
    return surface, (image, alpha), size_mm


def layer_b_marshal(
    image: np.ndarray | None,
    alpha: np.ndarray | None,
    size_mm: tuple[float, float],
    geom: RasterGeom,
) -> tuple[Part, bytes | None]:
    part = Part(
        size_mm=size_mm,
        pixels_per_mm=(geom.px_per_mm_x, geom.px_per_mm_y),
    )
    part.image_source = WholeImageSource(image)
    alpha_arr = (
        (alpha * 255).astype(np.uint8).tobytes() if alpha is not None else None
    )
    return part, alpha_arr


def layer_c_assemble(
    part: Part,
    alpha_arr: bytes | None,
    geom: RasterGeom,
    mode: str,
    num_power_levels: int,
    cache_budget_bytes: int,
) -> dict:
    spec = RasterSpec(
        mode=mode,
        line_interval_mm=geom.line_interval_mm,
        sample_interval_mm=geom.sample_interval_mm,
        min_power=0.0,
        max_power=1.0,
        step_power=0.1,
        num_power_levels=num_power_levels,
        angle=0.0,
        scan_mode="segmented",
        cross_hatch=False,
        num_depth_levels=5,
        alpha=alpha_arr,
    )
    node = NodeRequest(
        key="wp",
        generation_id=1,
        stage=StageSpec.Compute(
            part=part,
            params=ComputePayload(assembler=Assembler(spec)),
        ),
    )
    nodes = [
        node,
        _aggregate_node("step", ["wp"]),
        _aggregate_node("job", ["step"]),
    ]
    pipe = Pipeline(cache_budget_bytes)
    completed = []
    pipe.clear_cache()
    pipe.execute(nodes, completed.append, None)
    pipe.clear_cache()
    by_key = {c.key: c for c in completed}
    final = by_key.get("job") or by_key.get("wp")
    if final is None or final.error is not None:
        err = final.error if final else "no completion"
        raise RuntimeError(f"assembly failed: {err}")
    ops = final.output.ops
    info = {
        "ops_len": ops.len(),
        "ops_heap_mb": ops.heap_size() / 1e6,
    }
    return info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_one(
    src: pyvips.Image,
    spot_mm: float,
    size_mm: tuple[float, float],
    mode: str,
    num_power_levels: int,
    sampler: Sampler,
    cache_budget_bytes: int,
    verbose: bool = False,
) -> dict:
    geom = compute_geom(spot_mm, size_mm)

    with sampler.phase("A: render + preprocess"):
        _surface, (image, alpha), size_mm = layer_a_preprocess(src, geom, mode)
    if image is None:
        raise RuntimeError("preprocess returned no image")

    with sampler.phase("B: PyO3 marshalling"):
        part, alpha_arr = layer_b_marshal(image, alpha, size_mm, geom)

    with sampler.phase("C: Rust assembly"):
        ops_info = layer_c_assemble(
            part,
            alpha_arr,
            geom,
            mode,
            num_power_levels,
            cache_budget_bytes,
        )

    if verbose:
        snap = tracemalloc.take_snapshot()
        print("\nTop Python allocators (this run, cumulative):")
        for stat in snap.statistics("lineno")[:12]:
            print(f"  {stat}")

    a = sampler.phases["A: render + preprocess"]
    b = sampler.phases["B: PyO3 marshalling"]
    c = sampler.phases["C: Rust assembly"]
    return {
        "spot_mm": spot_mm,
        "target": f"{geom.target_w}x{geom.target_h}",
        "mpx": geom.target_w * geom.target_h / 1e6,
        "A_time": a.elapsed,
        "A_native_mb": a.peak_arena_delta / 1e6,
        "A_py_mb": a.peak_traced_delta / 1e6,
        "A_rss_mb": a.peak_rss_delta / 1e6,
        "B_time": b.elapsed,
        "B_native_mb": b.peak_arena_delta / 1e6,
        "B_py_mb": b.peak_traced_delta / 1e6,
        "B_rss_mb": b.peak_rss_delta / 1e6,
        "C_time": c.elapsed,
        "C_native_mb": c.peak_arena_delta / 1e6,
        "C_py_mb": c.peak_traced_delta / 1e6,
        "C_rss_mb": c.peak_rss_delta / 1e6,
        "ops_len": ops_info["ops_len"],
        "ops_heap_mb": ops_info["ops_heap_mb"],
    }


def parse_size(s: str) -> tuple[float, float]:
    w, h = s.lower().split("x")
    return float(w), float(h)


def print_row(cols, widths):
    print("  ".join(str(c).rjust(w) for c, w in zip(cols, widths)))


def print_table(rows: list[dict]):
    headers = [
        "spot",
        "target",
        "MP",
        "A_s",
        "A_MB",
        "B_s",
        "B_MB",
        "C_s",
        "C_MB",
        "ops_len",
        "ops_MB",
    ]
    widths = [6, 11, 6, 6, 8, 6, 8, 6, 8, 10, 9]
    print_row(headers, widths)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print_row(
            [
                f"{r['spot_mm']:.3f}",
                r["target"],
                f"{r['mpx']:.1f}",
                f"{r['A_time']:.2f}",
                f"{r['A_native_mb']:.0f}",
                f"{r['B_time']:.2f}",
                f"{r['B_native_mb']:.0f}",
                f"{r['C_time']:.2f}",
                f"{r['C_native_mb']:.0f}",
                r["ops_len"],
                f"{r['ops_heap_mb']:.0f}",
            ],
            widths,
        )
    note = (
        "  _MB columns = peak native-heap delta per layer (mallinfo2).\n"
        "  Run under `memray run --native` for the authoritative\n"
        "  process-wide transient peak (incl. mmap / non-heap)."
    )
    print(note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="Path to a PNG raster image")
    ap.add_argument(
        "--spot-mm",
        type=float,
        default=0.1,
        help="Laser spot size in mm (default 0.1)",
    )
    ap.add_argument(
        "--size-mm",
        type=parse_size,
        default=(200.0, 260.0),
        help="Workpiece size as WxH mm (default 200x260)",
    )
    ap.add_argument(
        "--mode",
        default="power_modulated",
        choices=sorted(DEPTH_BY_RAYGEO_NAME),
        help="Raster depth mode (default power_modulated)",
    )
    ap.add_argument(
        "--num-power-levels",
        type=int,
        default=10,
        help="Power quantization levels (default 10)",
    )
    ap.add_argument(
        "--sweep",
        default=None,
        help="Comma-separated spot sizes in mm, e.g. 0.05,0.1,0.2,0.4",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Print top Python allocators"
    )
    ap.add_argument(
        "--cache-budget-gb",
        type=float,
        default=16.0,
        help="raygeo cache budget in GiB (default 16)",
    )
    args = ap.parse_args(argv)

    tracemalloc.start(25)
    sampler = Sampler()
    sampler.start()
    try:
        src = load_source_image(args.image)
        print(
            f"image: {args.image}  "
            f"{src.width}x{src.height} "
            f"({src.width * src.height / 1e6:.1f} MP, {src.bands} bands)"
        )
        print(
            f"mode: {args.mode}  size: {args.size_mm[0]}x"
            f"{args.size_mm[1]} mm\n"
        )

        if args.sweep:
            spots = [float(x) for x in args.sweep.split(",")]
        else:
            spots = [args.spot_mm]

        rows = []
        budget = int(args.cache_budget_gb * 1024**3)
        for sp in spots:
            sampler.phases.clear()
            r = run_one(
                src,
                sp,
                args.size_mm,
                args.mode,
                args.num_power_levels,
                sampler,
                budget,
                args.verbose,
            )
            rows.append(r)
        print_table(rows)
    finally:
        sampler.stop()
        tracemalloc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
