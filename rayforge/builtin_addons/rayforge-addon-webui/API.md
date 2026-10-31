# Rayforge Web UI API

The `webui` builtin addon serves this API from `rayforge serve` under
the prefix `/api/v1`. Every request requires
`Authorization: Bearer <token>` when the server was started with
`--token`; WebSocket clients may instead pass `?token=<token>`.

## Authentication

| Transport | Mechanism |
| --- | --- |
| HTTP | `Authorization: Bearer <token>` header |
| WebSocket | Same header, or `?token=<token>` query parameter |

Without `--token` no authentication is required (loopback setups).

## Coordinate contract

Rayforge spaces: `WORLD` (document, mm, bottom-left origin, Y-up),
`MACHINE` (native bed coordinates; origin corner and axis directions
per machine, plus optional reverse_x/reverse_y sign flips), `WORKAREA`
(margins inset), `COMMAND` (G-code output, WCS-relative), `PIXEL`
(raster, Y-down), and `PANEL` (WORLD rotated by the panel orientation:
`native`, `rotated_left`, `rotated_right`).

Transformation chains:

- **Document → Screen**: WORLD → panel rotation
  (`GET /viewport` `world_to_panel`, a rigid 2x3 matrix) → the
  frontend maps PANEL X/Y onto its ground plane (three.js X/Z, Y-up).
- **Document → Machine → Gcode**: world ops → MachineTransformSpec
  (raygeo) → MACHINE coords → minus the active WCS offset → COMMAND.
- **Machine report → Screen**: DeviceState → `Machine.get_beam_position()`
  → `panel.machine_point_to_panel()` → `viewport.beam`.

The panel coordinates in this API follow `x_axis_right` /
`y_axis_down`: the displayed origin corner determines axis polarity.
The server performs every conversion through `MachineSpace` /
`MachinePanel` (`webui/coordinates.py`); the frontend must never
re-derive them.

`GET /viewport` fields:

| Field | Meaning |
| --- | --- |
| `orientation` | `native` / `rotated_left` / `rotated_right` |
| `world_to_panel` | rigid WORLD→PANEL 2x3 matrix `[[a,b,tx],[d,e,ty]]` |
| `x_axis_right` | displayed origin on the right (X grows left) |
| `y_axis_down` | displayed origin at the top (Y grows down) |
| `x_axis_negative`, `y_axis_negative` | native axis reversal shown on that displayed axis |
| `extents`, `margins` | bed size `[w, h]` and margins `[l, t, r, b]` in PANEL mm |
| `workarea` | work-area rect `[x, y, w, h]` in PANEL mm |
| `workarea_origin` | work-area origin in PANEL mm (the corner the axis arrows anchor to) |
| `work_origin` | active work origin (WCS, or the workarea origin in workarea-origin mode) in PANEL mm — where the "W" marker is drawn |
| `soft_limits` | soft limits `[x_min, y_min, x_max, y_max]` in MACHINE mm |
| `soft_limits_panel` | same rect as `[x, y, w, h]` in PANEL mm |
| `nogo_zones` | enabled no-go zones, each a closed PANEL-space polygon `[x, y, x, y, ...]` (rect corners, or a sampled circle for cylindrical zones); empty list when the machine has none |
| `rotary_active` | rotary module(s) configured (frontend warning banner) |
| `wcs` | `{name, offset_panel, offset_machine}` or `null` when the work origin is the workarea origin |
| `beam` | beam position `[x, y]` in PANEL mm, or `null` when unknown |

## Endpoints

### System

| Method | Path | Description |
| --- | --- | --- |
| GET | `/system/info` | name, version, API version, machine count |
| GET | `/viewport` | the viewport contract (see above) |

### Document

| Method | Path | Description |
| --- | --- | --- |
| GET | `/doc/tree` | full tree: layers, groups, workpieces, steps; carries `rev`, a monotonic document revision |
| GET | `/doc/history` | `can_undo`, `can_redo`, `undo_labels`, `redo_labels`, `dirty`; `undo_labels` runs oldest→newest, `redo_labels` is in replay order (next redo first) |
| POST | `/doc/undo` | undo last command |
| POST | `/doc/redo` | redo last undone command |
| POST | `/doc/new` | fresh document (3 layers), clears history |

### Layout

| Method | Path | Description |
| --- | --- | --- |
| GET | `/layouts` | registered layout strategies: `[{id, label}]` |
| POST | `/layout` | body `{strategy, params?}` — run the strategy on the active layer's content items (undoable) → `{moved}`; 404 for an unknown strategy, 400 for an empty layer or an unknown param |

`/project/open` and `/doc/new` return the same tree shape (including
`rev`).

Tree entries: layers carry `uid, name, visible, color, wcs,
rotary_enabled, rotary_diameter_mm, rotary_module_uid, order,
workflow_uid, step_uids, content_uids`; workpieces carry `uid, name,
layer_uid, pos_mm [x,y], size_mm [w,h], angle_deg, geometry_revision,
transform_revision` (world-space decomposition — no raw matrices);
steps carry `uid, step_type, typelabel, name, params, workpiece_uids,
order, visible, selected_head_uid, layer_uid`.

The tree also carries `stock_items`: stock material instances living
at the document root, shaped like workpieces but with `uid, type:
"stock", name, visible, color (effective material color or null),
pos_mm, size_mm, angle_deg, geometry_revision, transform_revision`.
`GET /items/{uid}/geometry` serves their pristine world-space
geometry exactly like workpiece geometry.

### Items

| Method | Path | Description |
| --- | --- | --- |
| GET | `/items/{uid}/geometry?tolerance=` | world polylines + `outline [x,y,w,h]`, `rev` (workpieces and stock items) |
| GET | `/items/{uid}/texture.png?rev=` | PNG raster of the workpiece source (ETag = revision) |
| PUT | `/items/{uid}/transform` | body `{pos_mm?, size_mm?, angle_deg?}` (WORLD mm) — undoable |
| POST | `/items/delete` | body `{uids: [...]}` |
| POST | `/items/duplicate` | body `{uids: [...]}` → `{created: [...]}` |
| POST | `/items/flip` | body `{uids, axis: horizontal\|vertical}` |
| POST | `/items/align` | body `{uids, mode}`: `left, right, top, bottom, center_horizontal, center_vertical` |
| POST | `/items/distribute` | body `{uids, axis: horizontal\|vertical}` |

### Layers

| Method | Path | Description |
| --- | --- | --- |
| GET | `/layers` | list + active layer uid |
| GET | `/layers/{uid}` | single layer |
| POST | `/layers` | add layer (becomes active) |
| PUT | `/layers/{uid}` | body `{name?, visible?, color?, wcs?, rotary_enabled?}` |
| DELETE | `/layers/{uid}` | delete (400 on the last layer) |
| POST | `/layers/{uid}/activate` | set active layer |
| POST | `/layers/reorder` | body `{uids: [...]}` full ordering |

### Steps

| Method | Path | Description |
| --- | --- | --- |
| GET | `/steps` | registered step types with VarSet form schemas |
| GET | `/steps/{uid}/schema` | schema groups of a step instance |
| GET | `/layers/{uid}/steps` | steps of one layer |
| POST | `/layers/{uid}/steps` | body `{step_type, name?, params?}` |
| PUT | `/steps/{uid}` | body `{name?, params?}` — undoable |
| DELETE | `/steps/{uid}` | remove step |

Schema fields: `{key, type: boolean\|number\|string, kind: input,
switch, slider, length, speed, angle, textarea, label, description,
default, min, max, step, options [{value,label}], unit: mm\|mm/min\|deg,
optional}`.

### Import / project

| Method | Path | Description |
| --- | --- | --- |
| POST | `/import/analyze` | multipart `file` → importer id, features, action (`direct_load` / `interactive_config` / `unsupported`), scan manifest |
| POST | `/import` | multipart `file`; optional query `spec` (JSON `VectorizationSpec` dict, e.g. `{"type": "PassthroughSpec"}` or `{"type": "TraceSpec"}`), `x`, `y` (WORLD mm) → `{created: [uids], layer_uid}` |
| POST | `/project/open` | multipart `.ryp` file → new document tree |
| GET | `/project/save` | `.ryp` download of the current document |
| GET | `/project/gcode` | generated gcode of the current job artifact (404 without one) |

### Machines

| Method | Path | Description |
| --- | --- | --- |
| GET | `/machines` | id, name, driver, connected, active, placeholder |
| POST | `/machines/{id}/connect` | connect the driver |
| POST | `/machines/{id}/disconnect` | disconnect the driver |
| GET | `/machine/state` | live state of the active machine; `beam_position_panel` is the reported beam position in PANEL mm (null when unknown) |
| GET | `/drivers` | registered drivers: `[{name, maturity, maturity_warning}]` |
| GET | `/machines/{id}/driver-setup` | the driver's setup form: `{driver, setup: {title, fields}, values (merged defaults + `driver_args`), precheck_error, device_settings}`; `device_settings` lists the driver's live device ($$) VarSet definitions when the driver supports settings (read-only via this API), else `null` |
| PUT | `/machines/{id}/driver-setup` | body `{values: {...}}` (subset of the setup vars); validates keys against the driver's `get_setup_vars`, merges into `driver_args`, rebuilds the driver via the controller, returns the refreshed payload; unknown keys → 404 |
| PUT | `/machines/{id}/driver` | body `{driver}` — switch the driver class (registry name, see `/drivers`); resets `driver_args` to `{}`, syncs the dialect, rebuilds; returns the refreshed driver-setup payload |
| GET | `/machines/{id}/settings` | editable machine fields: `name`, `axis_extents [w,h]`, `origin` (`bottom_left`…), `panel_orientation`, `work_margins [l,t,r,b]`, `reverse_x_axis`, `reverse_y_axis`, `reverse_z_axis`, `soft_limits_enabled`, `max_travel_speed`, `max_cut_speed`, `acceleration`, `jog_speed`, `dialect_uid`, `heads`, `rotary_modules`, `rotary_enabled_default`, `nogo_zones` (full Zone dicts incl. `uid`), `coordinate_systems [{name, label, offset [x,y,z]}]` |
| PUT | `/machines/{id}/settings` | body `{values: {...}}` with a subset of the scalar geometry/motion fields; setters run on the machine (autosave + viewport/machine event push), unknown fields or values → 404/400; returns the full settings payload |
| PUT | `/machines/{id}/heads` | body `{heads: [{uid, ...patch}]}` — laser head settings (`max_power`, `frame_power` 0-1, `focus_power` 0-1, `frame_speed`, `frame_repeat_count`, `frame_corner_pause`, `spot_size_mm [w,h]`, `focal_distance`, `pointer_offset_enabled/x/y`); unknown uid → 404; returns the heads payload |
| GET | `/machines/{id}/macros` | `{macros: [{uid, name, code: [lines], enabled}]}` |
| POST | `/machines/{id}/macros` | body `{name, code (string, newline-separated), enabled?}` → created macro |
| PUT | `/machines/{id}/macros/{uid}` | body `{name?, code?, enabled?}` → updated macro (autosaved) |
| DELETE | `/machines/{id}/macros/{uid}` | remove a macro |
| PUT | `/machines/{id}/wcs/{slot}` | body `{offset: [x,y,z], label?}` — update a work offset (autosaved); unknown slot → 404 |
| POST | `/machines/{id}/nogo-zones` | body `{name?, shape: rect\|cylinder, params: {x, y, w, h}` or `{x, y, radius}`, `enabled?}` → created Zone dict; viewport contract and machine settings push updates automatically |
| DELETE | `/machines/{id}/nogo-zones/{uid}` | remove a no-go zone |
| PUT | `/machines/{id}/rotary-modules/{uid}` | body `{name?, default_diameter?, mm_per_rotation?, reverse_axis?}` → updated rotary-modules payload (unknown uid → 404) |

## Settings

Tabbed preferences dialog backing (`General | Units | Import | Advanced`
in the web UI).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/settings` | editable app config: `theme` (`system\|light\|dark`), `language` (ISO code or null = system default; applies after a page reload), `import_dpi`, `cache_budget_bytes`, `watch_source_files`, `check_for_app_updates`, `ops_color_mode` (`laser\|layer`), `unit_preferences` (`{length, speed, acceleration}`) |
| PUT | `/settings` | body `{values: {...}}` with a subset of those fields; validated and applied through the ConfigManager setters (autosaved); unknown keys or out-of-range values → 400; returns the full settings payload |

Not ported to the web UI (standalone managers in the GTK app, not
simple settings): Gestures, Materials, Recipes, Color presets,
Add-ons and Licenses/AI pages.

### Machine control

All panel-space inputs are converted through the coordinate contract.

| Method | Path | Description |
| --- | --- | --- |
| POST | `/machine/home` | body `{axes: ["x","y","z"]?}` |
| POST | `/machine/jog` | body `{direction: "+x"\|"-x"\|"+y"\|"-y"\|"+z"\|"-z", distance_mm, speed?}` → native deltas |
| POST | `/machine/move-to` | body `{x, y, z?, speed?}` — PANEL mm |
| POST | `/machine/set-power` | body `{head?, percent 0-1}` |
| POST | `/machine/set-focus-power` | body `{head?, percent 0-1}` |
| POST | `/machine/run-raw` | body `{gcode}` |
| POST | `/machine/clear-alarm` | clear active alarm |
| POST | `/machine/set-work-origin-here` | body `{axes?}` zero active WCS at current position |
| POST | `/machine/set-work-origin` | body `{x, y, z, wcs_slot?}` — PANEL mm |
| POST | `/machine/select-wcs` | body `{slot}` (e.g. `G54`) |
| POST | `/machine/sync-wcs` | re-read WCS offsets from the device |
| POST | `/machine/macros/{uid}/run` | run an enabled machine macro |
| POST | `/machine/frame` | body `{include_overscan?}` — frame the job |

### Job

| Method | Path | Description |
| --- | --- | --- |
| POST | `/job/start` | assemble and send the job (409 when one is running) |
| POST | `/job/hold` | body `{hold: bool}` pause/resume |
| POST | `/job/cancel` | cancel the running job |
| GET | `/job/status` | `{active, job}` with `kind, phase: running\|paused, progress_fraction, eta_seconds, ...` |

### Ops preview

| Method | Path | Description |
| --- | --- | --- |
| GET | `/ops?tolerance=` | world-space toolpath of the current job artifact: `{segments: [{points: [[x, y], ...], power: 0-1, rate, cutting: bool}], estimated_seconds, distance_mm, generation_id}` — 404 without an artifact; never triggers generation. `points` are WORLD mm (Y-up); `cutting: false` marks travel moves |

## WebSocket events

`WS /api/v1/ws`. On connect the server sends a snapshot: a `doc`
snapshot digest, a `machine` state event, and a `viewport` contract
event. Afterwards it pushes:

| `type` | Payload | Trigger |
| --- | --- | --- |
| `doc` | `{kind: structure\|updated\|transform\|history\|active-layer\|processing\|settled\|snapshot, uids: [...], revisions: {uid: rev, tree: rev}, ...}` | doc/history/editor signals; `revisions.tree` is the current document revision and lets clients skip refetches when it matches their `/doc/tree` `rev` |
| `viewport` | `{data: {...contract...}}` | machine changed, WCS updated |
| `machine` | `{data: {...machine state...}}` | state changes, throttled to ~5 Hz |
| `job` | `{data: {active, job\|null}}` | job started / progress / finished |
| `notify` | `{level: info\|warning\|error, message}` | editor notifications |

Client messages are ignored; failed sends drop the client.
