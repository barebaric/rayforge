# Driver Feature Gaps: Raster Angle & Overscan

This document describes the design for handling machine/driver features that
Rayforge supports but a particular driver cannot honor. The two concrete cases
that motivated this work are:

1. **Ruida cannot raster in arbitrary directions.** It only supports horizontal
   raster scan lines, so the raster producer's **scan angle** and **cross-hatch**
   options are meaningless on a Ruida machine.
2. **Ruida adds overscan automatically.** Unlike Grbl, the Ruida firmware
   applies its own overscan, so Rayforge's **Overscan** transformer would
   double it up.

Although both are "driver feature gaps," they must be resolved differently
(see below). The design adds a general, extensible mechanism so future gaps
follow the same pattern.

---

## Background you need before reading this

- **Production runs in a subprocess.** The workpiece pipeline executes in a
  background process that receives only serialized dictionaries (workpiece,
  producer, transformers, laser, and a `settings` dict). The main process
  injects machine/driver capability info into that `settings` dict before
  launching the worker — this is already how `machine_supports_arcs` /
  `machine_supports_curves` reach the subprocess (`workpiece_stage.py`,
  `prepare_task_settings`).
- **Drivers declare capabilities as boolean class attributes** on the `Driver`
  base class (e.g. `supports_settings`, `supports_probing`). This design
  follows that convention.
- **The encoder is the choke point.** `Machine.encode_ops` (in
  `machine/models/machine.py`) is the single function that turns an `Ops`
  object into machine code. It is on the path for **both** Send and Export,
  including the jog-widget Send path that otherwise bypasses the sanity
  checker.
- **Raster output is expressed as `SCAN_LINE` commands** (the primitive that
  the raygeo rasterize functions emit and that encoders linearize). This is
  what lets us detect angled raster cheaply at encode time.

---

## Why the two gaps are handled differently

|                                   | Overscan                                     | Raster angle                                                                                      |
| --------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| What happens if we just "drop" it | Output is **correct** (Ruida does it itself) | Output is **wrong** (silently flattening a 45° engraving to horizontal ruins the intended result) |
| Resolution                        | **Skip** (graceful)                          | **Block** (refuse to produce/send)                                                                |
| Blocks Send/Export?               | No                                           | Yes                                                                                               |

Silently forcing a non-horizontal raster to horizontal would be a surprising,
quality-destroying change. So raster angle is treated as a **hard rejection**,
while overscan is **silently skipped** because skipping it produces exactly the
intended behavior.

---

## Design principles

1. **Never mutate the project to "fix" a gap.** The user's stored params
   (`scan_angle`, `cross_hatch`, overscan `enabled`/`distance`) are preserved
   untouched. This is essential for **multi-machine portability**: a user may
   build a project on a Grbl machine (45° raster, overscan on) and later open
   it on a Ruida machine. The settings survive the round trip.
2. **Defense in depth.** Each gap is enforced at more than one layer so that no
   single code path can emit unsupported machine code.
3. **No expensive pre-processing.** The production hard-gate piggybacks on work
   the encoder is already doing, rather than adding a separate validation pass.
4. **Follow existing conventions.** Driver capability booleans on `Driver`;
   capability flags injected into the `settings` dict exactly like
   `machine_supports_arcs`.

---

## 1. Driver capability flags

Add two boolean class attributes to `Driver` (`machine/driver/driver.py`,
alongside the existing `supports_settings` etc.):

```python
supports_raster_angle: bool = True   # can raster in non-horizontal directions
native_overscan: bool = False        # firmware applies overscan itself
```

Defaults represent the common (Grbl-like) case. `RuidaDriver`
(`machine/driver/ruida/ruida_driver.py`) overrides both:

```python
supports_raster_angle = False
native_overscan = True
```

All other drivers inherit the defaults and need no changes.

---

## 2. Overscan — graceful skip

Because skipping Rayforge's overscan on Ruida produces the _correct_ output,
this never blocks the user. It is resolved silently in production and shown as
inert in the UI.

### Production

- In `WorkPiecePipelineStage.prepare_task_settings`
  (`pipeline/stage/workpiece_stage.py`), inject the flag into `settings`
  (next to the existing `machine_supports_arcs` injection):

  ```python
  settings["driver_native_overscan"] = self._machine.driver.native_overscan
  ```

- In `OverscanTransformer.run`
  (`post_processors/transformers/overscan_transformer.py`), early-return when
  the flag is set. The `settings` dict already reaches the transformer
  (`workpiece_compute.py` `_apply_transformers` passes it through to every
  `transformer.run(..., settings=settings)`):

  ```python
  def run(self, ops, workpiece=None, context=None, stock_geometries=None, settings=None):
      if not self.enabled or math.isclose(self.distance_mm, 0.0):
          return
      if settings and settings.get("driver_native_overscan"):
          return   # the driver/firmware adds overscan itself
      ops.apply_overscan(self.distance_mm)
  ```

### UI

`OverscanSettingsWidget` (already wired to `machine.changed`) becomes
**insensitive** with an explanatory banner when
`get_context().machine.driver.native_overscan` is true — e.g.
_"This machine adds overscan automatically; the setting has no effect."_
The stored `enabled`/`distance_mm` are preserved so switching back to a
Grbl machine restores the transformer's effect.

---

## 3. Raster angle — two-layer block

Raster angle is refused rather than silently flattened. There are two layers:
a UI gate that stops the user from sending/exporting an unsupported job, and a
production hard-gate that raises if asked to encode one.

### Layer A — UI gate: Send / Export buttons

When the active driver does not support raster angle and a visible step has a
non-zero angle or cross-hatch enabled, the **Send** and **Export G-code**
buttons are made insensitive, with a tooltip explaining why. This extends the
existing sensitivity logic in `mainwindow.py` `_update_actions_and_ui`.

To keep that logic decoupled from `Rasterizer` internals, each producer
declares whether its current params require a feature the driver lacks. Add a
classmethod to `OpsProducer` (`pipeline/producer/base.py`, no-op default) and
override it in `Rasterizer` (`producers/raster_producer.py`):

```python
# OpsProducer base — default: no gap
@classmethod
def requires_unsupported_features(cls, params: dict, driver) -> bool:
    return False

# Rasterizer override
@classmethod
def requires_unsupported_features(cls, params, driver) -> bool:
    if driver.supports_raster_angle:
        return False
    return params.get("scan_angle", 0) != 0 or bool(params.get("cross_hatch"))
```

`_update_actions_and_ui` iterates visible steps, looks up each step's producer
class via `producer_registry` from `step.opsproducer_dict["type"]`, reads
`step.opsproducer_dict["params"]`, and calls the classmethod against the live
driver. If any step reports a gap, Send and Export are disabled. This is cheap
(it reads the param dict; it does not need a generated `Ops`) so it can run on
every document/machine change.

### Layer B — UI controls: hidden unless already active

In `RasterSettingsWidget` (`widgets/raster_widget.py`,
`_build_raster_geometry_group`), the angle slider, direction preview, and
cross-hatch switch are **hidden** when the driver lacks raster angle support —
**unless the project already has a non-default value**. The visibility rule:

```python
show_angle_options = (
    driver.supports_raster_angle
    or params.get("scan_angle", 0) != 0
    or bool(params.get("cross_hatch"))
)
```

Rationale:

- A **fresh** raster step on Ruida has default params (0°, no cross-hatch), so
  the controls are hidden and the UI stays clean.
- A **project brought from a Grbl machine** (e.g. 45° angle) onto Ruida
  reveals the controls so the user can see the value and set it to 0° to
  unblock Send. An advisory banner explains the constraint.

The widget re-evaluates visibility on `machine.changed` and when its params
change.

### Layer C — production hard-gate: encoder raises inline

The authoritative guarantee lives in the encoder. Because raster output is
emitted as `SCAN_LINE` commands, a non-horizontal scan line _is_ the
unsupported feature. `RuidaEncoder._handle_scan_line`
(`machine/driver/ruida/ruida_encoder.py`) raises while it is already iterating
the command stream — **no separate validation pass, no pre-processing**:

```python
def _handle_scan_line(self, ops, idx, machine, binary, text):
    end = ops.endpoint(idx)
    if abs(end[1] - self.current_pos[1]) > 1e-6:
        raise UnsupportedOpsError(
            _("This machine cannot engrave raster lines at an angle. "
              "Set the raster angle to 0 degrees and disable cross-hatch.")
        )
    # ...existing linearization + emit logic...
```

Notes:

- Vector `LINE_TO` paths are **not** checked — Ruida supports arbitrary vector
  cuts. Only `SCAN_LINE` (the raster primitive) is checked, so the gate
  distinguishes raster from vector for free.
- Cross-hatch's second pass (at `scan_angle + 90°`) is a vertical scan line and
  is caught here too.
- This raise happens inside `Machine.encode_ops`, which is on the path for
  Send, Export, **and** the jog-widget Send bypass (`jog_widget.py` runs Send
  without the sanity check). So it is the true last-resort guarantee.

### New exception

Add `UnsupportedOpsError(Exception)` near the other driver exceptions
(`machine/driver/driver.py`, alongside `DriverPrecheckError`). It carries a
translatable message. Callers already surface encode failures as toasts
(`machine/cmd.py` for send, `doceditor/file_cmd.py` for export), so no new
error-reporting plumbing is required.

---

## Layering summary

|                         | Overscan                                                        | Raster angle                                                                               |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Resolution**          | Skip (correct output)                                           | Block (cannot silently flatten)                                                            |
| **Production**          | Transformer reads `settings["driver_native_overscan"]` -> no-op | Encoder raises `UnsupportedOpsError` on non-horizontal `SCAN_LINE`                         |
| **UI**                  | Overscan widget insensitive + banner                            | Send/Export insensitive + tooltip; angle/cross-hatch controls hidden unless already active |
| **Blocks Send/Export?** | No                                                              | Yes                                                                                        |
| **Project portable?**   | Yes (params preserved)                                          | Yes (params preserved; user sets 0° to unblock)                                            |

---

## File-by-file change list

| Concern                                          | File                                                                                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Driver capability flags                          | `machine/driver/driver.py` (add flags), `machine/driver/ruida/ruida_driver.py` (override)                                                      |
| Overscan flag injection                          | `pipeline/stage/workpiece_stage.py` (`prepare_task_settings`)                                                                                  |
| Overscan skip in production                      | `builtin_addons/rayforge-addon-post/post_processors/transformers/overscan_transformer.py`                                                      |
| Overscan UI (insensitive + banner)               | `builtin_addons/rayforge-addon-post/post_processors/widgets/overscan_widget.py`                                                                |
| Producer feature-gap declaration                 | `pipeline/producer/base.py` (base classmethod), `builtin_addons/rayforge-addon-laser/laser_essentials/producers/raster_producer.py` (override) |
| Raster UI gate (Send/Export sensitivity)         | `ui_gtk/mainwindow.py` (`_update_actions_and_ui`)                                                                                              |
| Raster UI controls (hide-unless-active + banner) | `builtin_addons/rayforge-addon-laser/laser_essentials/widgets/raster_widget.py`                                                                |
| Encoder hard-gate + exception                    | `machine/driver/driver.py` (`UnsupportedOpsError`), `machine/driver/ruida/ruida_encoder.py` (`_handle_scan_line`)                              |

---

## Implementation notes

- **`settings` propagation for overscan** is already in place:
  `prepare_task_settings` -> worker -> `compute_workpiece_artifact` ->
  `_apply_transformers` -> `transformer.run(..., settings=settings)`.
- **No project mutation.** At no point should `scan_angle`, `cross_hatch`, or
  overscan params be rewritten to "fix" a gap. Their values must round-trip
  across machine switches unchanged.
- **Extensibility.** Future driver gaps follow this template: add a flag on
  `Driver`, decide skip-vs-block, and either inject into `settings` (skip) or
  raise in the encoder (block). Producers with new constraints override
  `requires_unsupported_features` so the UI button gate picks them up
  automatically.
