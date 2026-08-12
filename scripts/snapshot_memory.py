"""Memory-ownership snapshot for rayforge.

Run via::

    pixi run rayforge wolf.ryp --uiscript scripts/snapshot_memory.py

The script waits for the document to settle (pipeline + scene compilation
+ view rendering all idle), then walks the live object graph from known
roots and reports, per owner, how many bytes it currently holds.

Rust-side sizes are obtained from native getters:
  - ``Ops.heap_size()``            — the Rust Vec<OpNode> + state heap
  - ``CompressedArray.compressed_size``  — the zstd-compressed payload
  - ``CompressedArray.uncompressed_size`` — the original decompressed size
  - ``Pipeline.cache_used_bytes``   — the raygeo LRU cache total

Python-side sizes use ``numpy.ndarray.nbytes``, ``len(bytes)``, and
``sys.getsizeof`` for wrapper objects.

A ``gc.get_objects()`` type-sweep is included as a cross-check: the sum
of all live ``Ops`` / ``CompressedArray`` / ``ndarray`` / ``bytes``
should be roughly attributable to the reported owners.  Discrepancies
point to untracked holders.
"""

import gc
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np
from raygeo.cnc.execution.specs import AggregateOutput
from raygeo.compressed_array import CompressedArray
from raygeo.ops import Ops

from rayforge.pipeline.artifact.job import JobArtifact
from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact

if TYPE_CHECKING:
    from gi.repository import Gtk

    from rayforge.core.doc import Doc
    from rayforge.doceditor.editor import DocEditor
    from rayforge.pipeline.artifact.store import ArtifactStore
    from rayforge.pipeline.encoder.base import EncodedOutput
    from rayforge.pipeline.pipeline import Pipeline
    from rayforge.pipeline.view.view_manager import ViewManager
    from rayforge.ui_gtk.mainwindow import MainWindow
    from rayforge.ui_gtk.sim3d.scene_presenter import ScenePresenter

logger = logging.getLogger("memsnapshot")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_seen_ids: set[int] = set()
_ops_seen: set[int] = set()


def _claim(obj: object) -> bool:
    """Return True if obj was not yet seen (and now claimed)."""
    oid = id(obj)
    if oid in _seen_ids:
        return False
    _seen_ids.add(oid)
    return True


def _obj_bytes(obj: object) -> int:
    """Best-effort byte size for a single object (no recursion)."""
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, CompressedArray):
        return obj.compressed_size
    if isinstance(obj, Ops):
        try:
            return obj.heap_size()
        except (RuntimeError, TypeError):
            return sys.getsizeof(obj)
    if isinstance(obj, str):
        return len(obj)
    return sys.getsizeof(obj)


def _walk_size(obj: object, max_depth: int = 3) -> int:
    """Sum ``_obj_bytes`` over obj and its direct referents up to
    *max_depth* levels, claiming each object once."""
    if not _claim(obj):
        return 0
    total = _obj_bytes(obj)
    if max_depth <= 0:
        return total
    try:
        refs = gc.get_referents(obj)
    except (RuntimeError, TypeError):
        refs = []
    for ref in refs:
        if ref is obj:
            continue
        if type(ref) in _SCALAR_TYPES:
            continue
        total += _walk_size(ref, max_depth - 1)
    return total


_SCALAR_TYPES: frozenset[type[object]] = frozenset(
    {
        type(None),
        int,
        float,
        bool,
        complex,
        str,
        type,
        frozenset,
        set,
    }
)


# ---------------------------------------------------------------------------
# Typed sweep (cross-check via gc.get_objects)
# ---------------------------------------------------------------------------


class TypeStats(TypedDict):
    """Aggregate count and byte total for one swept type."""

    count: int
    bytes: int


def _gc_type_sweep() -> dict[str, TypeStats]:
    """Count and size all live objects of key types."""
    gc.collect()
    stats: dict[str, TypeStats] = defaultdict(lambda: {"count": 0, "bytes": 0})
    for obj in gc.get_objects():
        if isinstance(obj, Ops):
            key = "Ops"
        elif isinstance(obj, CompressedArray):
            key = "CompressedArray"
        elif isinstance(obj, np.ndarray):
            key = "ndarray"
        elif isinstance(obj, (bytes, bytearray)):
            key = "bytes"
        else:
            continue
        stats[key]["count"] += 1
        stats[key]["bytes"] += _obj_bytes(obj)
    return dict(stats)


# ---------------------------------------------------------------------------
# Per-owner measurements
# ---------------------------------------------------------------------------


class OwnerReport:
    def __init__(self, name: str) -> None:
        self.name = name
        self.bytes: int = 0
        self.items: list[tuple[str, int]] = []

    def add(self, label: str, obj: object) -> int:
        sz = _obj_bytes(obj)
        self.bytes += sz
        self.items.append((label, sz))
        return sz


def _claim_compressed(r: OwnerReport, label: str, ca: CompressedArray) -> None:
    """Record *ca*'s compressed size unless it was already claimed."""
    if _claim(ca):
        sz = ca.compressed_size
        r.bytes += sz
        r.items.append((f"{label} (compressed)", sz))


def _claim_array(r: OwnerReport, label: str, arr: np.ndarray | None) -> None:
    """Record *arr*'s nbytes unless it was already claimed."""
    if isinstance(arr, np.ndarray) and _claim(arr):
        sz = arr.nbytes
        r.bytes += sz
        r.items.append((label, sz))


def _add_bytes(r: OwnerReport, label: str, data: bytes) -> None:
    """Record the size of a raw *bytes* payload."""
    sz = len(data)
    r.bytes += sz
    r.items.append((label, sz))


def _claim_ops(
    r: OwnerReport, label: str, attr: str, ops: Ops, seen: set[int]
) -> None:
    """Record an Ops heap size, deduplicated by id()."""
    if id(ops) in seen:
        return
    seen.add(id(ops))
    h = ops.heap_size()
    r.bytes += h
    r.items.append((f"{label}.{attr}.heap_size", h))


def _measure_encoded_output(
    r: OwnerReport, label: str, encoded: "EncodedOutput | None"
) -> None:
    """Record the encoded G-code text and op-map payloads."""
    if encoded is None:
        return
    text_sz = len(encoded.text)
    r.bytes += text_sz
    r.items.append((f"{label}.encoded.text", text_sz))
    op_map = encoded.op_map
    spans_sz = len(op_map.op_to_machine_code_bytes)
    lines_sz = len(op_map.machine_code_to_op_bytes)
    r.bytes += spans_sz + lines_sz
    label1 = (
        f"{label}.encoded.op_map.op_to_mc "
        f"({op_map.op_count} ops, {spans_sz} B)"
    )
    r.items.append((label1, spans_sz))
    label2 = (
        f"{label}.encoded.op_map.mc_to_op "
        f"({op_map.line_count} lines, {lines_sz} B)"
    )
    r.items.append((label2, lines_sz))


def _measure_artifact_store(store: "ArtifactStore") -> OwnerReport:
    r = OwnerReport(name="ArtifactStore")
    seen_ops: set[int] = set()
    for key, art in store._artifacts.items():
        label = f"{type(art).__name__}[{key}]"
        # Walk the artifact wrapper (non-Ops fields only)
        wrapper_sz = sys.getsizeof(art)
        r.bytes += wrapper_sz
        r.items.append((f"{label} (wrapper)", wrapper_sz))
        # Ops fields — deduplicate by id()
        if isinstance(art, JobArtifact):
            _claim_ops(r, label, "ops", art.ops, seen_ops)
            r.bytes += 1
            r.items.append(
                (
                    (
                        f"{label}.ops.commands "
                        f"(cutting={art.ops.count_cutting()}, "
                        f"travel={art.ops.count_travel()}, "
                        f"scanline={art.ops.count_scanline()})"
                    ),
                    1,
                )
            )
            if art.mapped_ops is not None:
                _claim_ops(r, label, "mapped_ops", art.mapped_ops, seen_ops)
            _measure_encoded_output(r, label, art.encoded_output)
        elif isinstance(art, WorkPieceArtifact):
            _claim_ops(r, label, "ops", art.ops, seen_ops)
    return r


def _measure_view_manager(vm: "ViewManager") -> OwnerReport:
    r = OwnerReport(name="ViewManager")
    for composite_id, entry in vm._view_entries.items():
        label = f"ViewEntry{composite_id}"
        bitmap = entry.bitmap
        bsz = bitmap.nbytes if isinstance(bitmap, np.ndarray) else 0
        r.bytes += bsz
        r.items.append((f"{label}.bitmap", bsz))
    return r


def _measure_scene_presenter(presenter: "ScenePresenter") -> OwnerReport:
    r = OwnerReport(name="ScenePresenter")
    art = presenter._compiled_artifact
    if art is None:
        r.items.append(("compiled_artifact", 0))
        return r
    wrapper_sz = sys.getsizeof(art)
    r.bytes += wrapper_sz
    r.items.append(("compiled_artifact (wrapper)", wrapper_sz))
    for i, vl in enumerate(art.vertex_layers):
        label = f"VertexLayer[{i}]"
        _claim_compressed(r, f"{label}.powered_verts", vl.powered_verts)
        _claim_compressed(r, f"{label}.powered_attrib", vl.powered_attrib)
        _claim_compressed(r, f"{label}.travel_verts", vl.travel_verts)
        _claim_compressed(r, f"{label}.zero_power_verts", vl.zero_power_verts)
        _claim_array(r, f"{label}.powered_cmd_offsets", vl.powered_cmd_offsets)
        _claim_array(r, f"{label}.travel_cmd_offsets", vl.travel_cmd_offsets)
    for i, tl in enumerate(art.texture_layers):
        label = f"TextureLayer[{i}]"
        _claim_compressed(r, f"{label}.power_texture", tl.power_texture)
        _claim_array(r, f"{label}.model_matrix", tl.model_matrix)
        _claim_array(r, f"{label}.cylinder_vertices", tl.cylinder_vertices)
    for i, ol in enumerate(art.overlay_layers):
        label = f"OverlayLayer[{i}]"
        _claim_compressed(r, f"{label}.positions", ol.positions)
        _claim_compressed(r, f"{label}.overlay_attrib", ol.overlay_attrib)
        _claim_array(r, f"{label}.cmd_offsets", ol.cmd_offsets)
    return r


def _measure_pipeline(pipeline: "Pipeline") -> OwnerReport:
    r = OwnerReport(name="Pipeline")
    # raygeo cache
    cache_bytes = pipeline._raygeo_pipeline.cache_used_bytes
    r.bytes += cache_bytes
    r.items.append(("raygeo cache_used_bytes (reported)", cache_bytes))
    # Last aggregate output (Rust object) — deduplicate with store
    agg = cast(AggregateOutput | None, pipeline._last_aggregate_output)
    if agg is not None:
        ops = agg.ops
        if id(ops) not in _ops_seen:
            _ops_seen.add(id(ops))
            h = ops.heap_size()
            r.bytes += h
            r.items.append(("_last_aggregate_output.ops.heap_size", h))
    return r


def _measure_source_assets(doc: "Doc") -> OwnerReport:
    r = OwnerReport(name="SourceAssets")
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            sa = wp.source
            if sa is None:
                continue
            label = f"SourceAsset[{sa.name}]"
            _add_bytes(r, f"{label}.original_data", sa.original_data)
            if sa.base_render_data is not None:
                _add_bytes(r, f"{label}.base_render_data", sa.base_render_data)
            if sa.thumbnail_data is not None:
                _add_bytes(r, f"{label}.thumbnail_data", sa.thumbnail_data)
            cache = sa._base_image_cache
            if cache:
                r.items.append(
                    (f"{label}._base_image_cache entries", len(cache))
                )
    return r


def _read_rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _read_smaps_rollup() -> dict[str, int] | None:
    """Read /proc/self/smaps_rollup for a detailed RSS breakdown."""
    try:
        result = {}
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        result[parts[0].rstrip(":")] = int(parts[1]) * 1024
                    except ValueError:
                        continue
        return result
    except OSError:
        return None


def _malloc_trim() -> None:
    """Release freed-but-cached arena pages back to the OS."""
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _read_mallinfo() -> dict[str, int] | None:
    """Read glibc's full malloc statistics.

    Returns a dict with these keys (all in bytes):

    - ``uordblks``  — in-use small/medium heap (sbrk arena)
    - ``fordblks``  — free small/medium heap (sbrk arena)
    - ``arena``     — total sbrk arena (uordblks + fordblks)
    - ``hblkhd``    — in-use large-block mmap'd allocations
    - ``hblks``     — count of mmap'd blocks
    - ``usmblks``   — in-use fastbin bytes
    - ``fsmblks``   — free fastbin bytes
    - ``keepcost``  — top-most releasable chunk

    The total glibc-managed in-use memory is ``uordblks + hblkhd``.
    ``uordblks`` alone misses all allocations > ~128 KB, which glibc
    services via ``mmap`` (tracked in ``hblkhd``), not ``sbrk``.
    """
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None

    def _fields(t):
        return [
            ("arena", t),
            ("ordblks", t),
            ("smblks", t),
            ("hblks", t),
            ("hblkhd", t),
            ("usmblks", t),
            ("fsmblks", t),
            ("uordblks", t),
            ("fordblks", t),
            ("keepcost", t),
        ]

    try:
        libc.mallinfo2.restype = type(
            "_MI", (ctypes.Structure,), {"_fields_": _fields(ctypes.c_size_t)}
        )
        mi = libc.mallinfo2()
    except AttributeError:
        try:
            libc.mallinfo.restype = type(
                "_MI", (ctypes.Structure,), {"_fields_": _fields(ctypes.c_int)}
            )
            mi = libc.mallinfo()
        except AttributeError:
            return None

    return {
        "uordblks": int(mi.uordblks),
        "fordblks": int(mi.fordblks),
        "arena": int(mi.arena),
        "hblkhd": int(mi.hblkhd),
        "hblks": int(mi.hblks),
        "usmblks": int(mi.usmblks),
        "fsmblks": int(mi.fsmblks),
        "keepcost": int(mi.keepcost),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def _print_report(
    owners: list[OwnerReport], sweep: dict[str, TypeStats], rss_kb: int
) -> None:
    print("\n" + "=" * 72)
    print(" MEMORY OWNERSHIP SNAPSHOT (after pipeline settle)")
    print("=" * 72)

    total_attributed = 0
    for r in sorted(owners, key=lambda x: -x.bytes):
        print(f"\n● {r.name}: {_format_bytes(r.bytes)}")
        total_attributed += r.bytes
        for label, sz in sorted(r.items, key=lambda x: -x[1]):
            if sz > 0:
                print(f"    {label:55s} {_format_bytes(sz):>12s}")

    mi = _read_mallinfo()
    smaps = _read_smaps_rollup()

    print(f"\n{'─' * 72}")
    print(f"  Total attributed:  {_format_bytes(total_attributed):>12s}")
    print(f"  Process RSS:        {_format_bytes(rss_kb * 1024):>12s}")
    if mi is not None:
        glibc_inuse = mi["uordblks"] + mi["hblkhd"]
        glibc_free = mi["fordblks"] + mi["fsmblks"]
        print(f"  glibc in-use (sbrk):  {_format_bytes(mi['uordblks']):>12s}")
        print(f"  glibc in-use (mmap):   {_format_bytes(mi['hblkhd']):>12s}")
        print(f"  glibc in-use (total):  {_format_bytes(glibc_inuse):>12s}")
        print(f"  glibc free (wasted):   {_format_bytes(glibc_free):>12s}")
    if smaps:
        print("  smaps_rollup:")
        for k in ("Rss", "Pss", "Anonymous", "Swap", "File"):
            if k in smaps:
                print(f"    {k:20s} {_format_bytes(smaps[k]):>12s}")

    print(f"\n{'─' * 72}")
    print(" GC type sweep (all live objects, cross-check):")
    print("  (Note: Ops and CompressedArray are PyO3 objects and do NOT")
    print("   appear in gc.get_objects(); per-owner measurement above")
    print("   uses native heap_size()/compressed_size getters.)")
    for key in ("Ops", "CompressedArray", "ndarray", "bytes"):
        s = sweep.get(key, {"count": 0, "bytes": 0})
        print(
            f"  {key:20s}  count={s['count']:>8d}  "
            f"total={_format_bytes(s['bytes']):>12s}"
        )

    gap = rss_kb * 1024 - total_attributed
    print(f"\n  RSS − attributed gap: {_format_bytes(gap)}")
    if mi is not None:
        glibc_inuse = mi["uordblks"] + mi["hblkhd"]
        glibc_free = mi["fordblks"] + mi["fsmblks"]
        glibc_total = glibc_inuse + glibc_free
        heap_vs_attr = glibc_inuse - total_attributed
        non_glibc = rss_kb * 1024 - glibc_total
        print(f"  glibc in-use − attributed: {_format_bytes(heap_vs_attr)}")
        print(f"  glibc free (wasted):        {_format_bytes(glibc_free)}")
        print(f"  RSS − glibc total (non-glibc): {_format_bytes(non_glibc)}")
    print("  (non-glibc = Python interpreter, GL/GTK textures, thread")
    print("   stacks, pyvips image caches, and other non-malloc memory)")
    print("=" * 72 + "\n")


class AppProtocol(Protocol):
    """Minimal application surface this script depends on."""

    def quit_idle(self) -> None: ...


def _find_scene_presenter(win: "MainWindow") -> "ScenePresenter | None":
    """Locate the ScenePresenter on the 3D canvas, if it exists."""
    try:
        from rayforge.ui_gtk.sim3d.canvas3d import Canvas3D

        def search(widget: "Gtk.Widget") -> "ScenePresenter | None":
            if isinstance(widget, Canvas3D):
                return widget._presenter
            child = widget.get_first_child()
            while child is not None:
                found = search(child)
                if found is not None:
                    return found
                child = child.get_next_sibling()
            return None

        return search(win)
    except (RuntimeError, TypeError):
        return None


def _wait_for_settle(
    editor: "DocEditor", quiet_seconds: float = 2.0, timeout: float = 300.0
) -> bool:
    """Block until ``editor.is_processing`` has been False for
    *quiet_seconds* consecutive seconds, or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    last_busy = None
    idle_since = None
    while time.monotonic() < deadline:
        busy = editor.is_processing
        if busy:
            idle_since = None
        else:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= quiet_seconds:
                return True
        if busy != last_busy:
            logger.debug(
                "settle-wait: is_processing=%s idle_since=%s",
                busy,
                idle_since,
            )
            last_busy = busy
        time.sleep(0.2)
    logger.warning("settle-wait timed out after %.0fs", timeout)
    return False


def _switch_to_3d_view(win: "MainWindow") -> None:
    """Switch the view stack to the 3D page so the GLArea realizes."""
    try:
        win.view_stack.set_visible_child_name("3d")
        logger.info("snapshot_memory: switched to 3D view")
    except (AttributeError, RuntimeError) as e:
        logger.warning("snapshot_memory: failed to switch to 3D: %s", e)


def _wait_for_gl(win: "MainWindow", timeout: float = 30.0) -> None:
    """Wait for the 3D canvas GL to initialize and scene to compile."""
    canvas = win.canvas3d
    if canvas is None:
        logger.warning("snapshot_memory: no canvas3d on win")
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if canvas._gl_initialized:
            logger.info("snapshot_memory: GL initialized")
            return
        time.sleep(0.3)
    logger.warning(
        "snapshot_memory: GL did not initialize within %.0fs", timeout
    )


def _wait_for_scene_compiled(
    win: "MainWindow", timeout: float = 120.0
) -> None:
    """Wait for the ScenePresenter to have a compiled artifact."""
    canvas = win.canvas3d
    if canvas is None:
        return
    presenter = canvas._presenter
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if presenter._compiled_artifact is not None:
            logger.info("snapshot_memory: scene compiled")
            return
        time.sleep(0.5)
    logger.warning(
        "snapshot_memory: scene did not compile within %.0fs", timeout
    )


def run_snapshot(app: AppProtocol, win: "MainWindow") -> None:
    """Entry point — called from the UI script thread."""
    logger.info("snapshot_memory: waiting for document to settle...")
    editor = win.doc_editor
    _wait_for_settle(editor, quiet_seconds=3.0, timeout=300.0)

    # Switch to 3D view so the GLArea realizes and the scene compiles.
    _switch_to_3d_view(win)
    _wait_for_gl(win, timeout=30.0)
    _wait_for_scene_compiled(win, timeout=120.0)
    # Wait again for any extra work triggered by the scene compilation.
    _wait_for_settle(editor, quiet_seconds=3.0, timeout=60.0)

    logger.info("snapshot_memory: gc.collect() before snapshot")
    gc.collect()
    _malloc_trim()
    time.sleep(0.5)

    pipeline = editor.pipeline
    store = pipeline.artifact_store
    vm = editor.view_manager
    doc = editor.doc
    presenter = _find_scene_presenter(win)

    global _seen_ids
    _seen_ids = set()
    _ops_seen.clear()

    owners = [
        _measure_pipeline(pipeline),
        _measure_artifact_store(store),
        _measure_view_manager(vm),
        _measure_source_assets(doc),
    ]
    if presenter is not None:
        owners.append(_measure_scene_presenter(presenter))
    else:
        owners.append(OwnerReport(name="ScenePresenter (no 3D canvas)"))

    sweep = _gc_type_sweep()
    rss = _read_rss_kb()

    def _force_exit() -> None:
        time.sleep(5)
        logger.warning("snapshot_memory: force exit after 5s")
        os._exit(0)

    threading.Thread(target=_force_exit, daemon=True).start()

    try:
        _print_report(owners, sweep, rss)
    except Exception:
        logger.exception("snapshot_memory: report printing failed")

    logger.info("snapshot_memory: done, quitting app.")
    app.quit_idle()


# ── UI script entry point ──────────────────────────────────────────
# When run via --uiscript, the globals `app` and `win` are injected
# by rayforge.uiscript._set_context().
_app: AppProtocol | None = None
_win: "MainWindow | None" = None
try:
    from rayforge import uiscript as _ui

    _app = _ui.app
    _win = _ui.win
except Exception:  # noqa: BLE001, S110
    pass

if _app is not None and _win is not None:
    t = threading.Thread(target=run_snapshot, args=(_app, _win), daemon=True)
    t.start()
else:
    # Allow direct invocation for testing
    if __name__ == "__main__":
        print(
            "This script must be run via: rayforge <file> --uiscript <script>"
        )
        sys.exit(1)
