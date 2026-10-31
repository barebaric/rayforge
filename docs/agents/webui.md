# Web UI (rayforge-addon-webui)

The `rayforge serve` command boots a GTK-free rayforge environment and
serves a web UI from the `webui` builtin addon
(`rayforge/builtin_addons/rayforge-addon-webui/`). The addon is a
FastAPI app (`webui/`) plus a three.js SPA (`frontend/`); the built
bundle in `webui/static/` is committed.

## Commands

```bash
pixi run python -m rayforge serve --config /tmp/rf-x --port 8080
pixi run pytest rayforge/builtin_addons/rayforge-addon-webui/tests -q
pixi run web-test        # frontend vitest suite
pixi run web-build       # rebuild frontend/webui/static (commit the output)
python rayforge/builtin_addons/rayforge-addon-webui/scripts/gen_oriented_configs.py /tmp/rf-orient  # one config dir per machine-orientation variant; serve with --config /tmp/rf-orient/<variant>
```

The HTTP and WS API contract lives in the addon's `API.md`; keep it in
sync with `webui/` when behavior changes.

## Coordinate contract

All server-side conversion goes through `MachineSpace` /
`MachinePanel` (`webui/coordinates.py`); never re-derive the math.
Rules that keep the layers consistent:

- Router payloads that describe bed positions (jog targets,
  move-to, set-work-origin) are PANEL space; convert them at the
  router boundary, not in the frontend.
- `GET /viewport` is the single source of truth for the frontend
  (`orientation`, `world_to_panel`, extents, workarea, soft limits,
  WCS, beam). The frontend maps PANEL X/Y onto its three.js ground
  plane (X/Z, Y-up) via `frontend/src/scene/coords.ts` — that module
  is the only place allowed to do this mapping.
- Workpiece transforms in the API are WORLD-space decompositions
  (`pos_mm`, `size_mm`, `angle_deg`); raw matrices never cross the API.
- Machine reports reach the frontend as PANEL beam positions via
  `Machine.get_beam_position()` → `machine_point_to_panel()`.

## Conventions

- Backend user-visible strings are wrapped with `_()`; the frontend
  has no i18n catalog yet (see follow-ups).
- Document mutations must go through editor commands that create
  history entries (e.g. `editor.layout` helpers already record undo);
  the layout strategies run asynchronously, so HTTP responses return
  before the deltas are applied — the WS `doc` events settle the UI.
- The frontend fetches `/doc/tree` `rev` and matches it against the
  `revisions["tree"]` value of WS doc events to decide whether a
  refetch is needed; bump both from `EventSource` only.

## Known deferred follow-ups

- Frontend i18n (no gettext catalog for the SPA).
- Camera capture preview in the canvas.
- Rotary cylinder view.
- Machine configuration UI (machines are read-only in the web UI).
- Multi-session support (one shared document; last writer wins).
