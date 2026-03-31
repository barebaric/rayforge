# Plan: Model-Based Rotary Accessories

## glTF and Units

There is no viable pure-Python STEP loader. STEP requires OpenCASCADE (heavy
C++ dependency). We stay with GLB via trimesh and require models to be authored
in millimeters (the codebase's convention). `mesh.bounds` from trimesh gives
the bounding box in the model's native units — if models are in mm, you get
physical size for free. For models not authored in mm, the scale component of
the 4×4 transform matrix handles it.

## 1. Simplify `RotaryModule`

**File:** `rayforge/machine/models/rotary_module.py`

**Remove** these parametric-only attributes, setters, serialization keys, and
`from_dict` restoration:

- `chuck_diameter`, `tailstock_diameter`, `chuck_height`, `tailstock_height`
- `max_height` — replace collision bbox with computation from model bounds
  (see below)
- `length`
- `viz_mode` (and the `RotaryVisualizationMode` enum)
- `viz_scale`, `viz_rotation` — subsumed by transform matrix

**Replace** `x`, `y`, `z` with a single `transform: np.ndarray` (4×4, default
identity). The 4×4 matrix covers translation, rotation, and scale. The UI
initially only exposes translation rows.

**Rename** `model_path: Optional[Path]` to `model_id: Optional[str]`. This is
an opaque identifier that only `ModelManager.resolve()` interprets. Could be a
library-relative path, a UUID, or any future scheme — the module doesn't care.

**Keep:**

- `uid`, `name`, `axis` — gcode axis selection
- `default_diameter` — layer default workpiece diameter for rotary gcode
- `extra: Dict[str, Any]` — forward compatibility

**Update `get_collision_bbox()`** — return `None` when no model is set. When a
model is set, compute from `ModelRenderer.bounds` transformed by
`module.transform`. This may need to be lazy/cached since it requires loading
the mesh.

**Update** `to_dict()` / `from_dict()`. The 4×4 matrix serializes as a flat
list of 16 floats.

## 2. Create a reusable `ModelSelectionDialog`

**New file:** `rayforge/ui_gtk/shared/model_selection_dialog.py`

A picker dialog that:

- Lists models from `ModelManager` libraries (reuse the same `ModelManager`
  API used by `ModelManagerPage`)
- Shows a 3D preview via `ModelPreviewWidget`
- Returns the selected `model_id` string (or `None` on cancel)

Extracts the browse/display logic from `ModelManagerPage` into a reusable
dialog.

## 3. Enhance the existing `RotaryModulePage`

**File:** `rayforge/ui_gtk/machine/rotary_module_page.py`

In the per-module properties group, **replace** all parametric rows (chuck
diameter, tailstock diameter, length, max height, chuck height, tailstock
height, X/Y/Z position) with:

- **Model** — an `Adw.ActionRow` that opens `ModelSelectionDialog` on click,
  displays the selected model name or "None"
- **Position X/Y/Z** — `Adw.SpinRow`s that write into the translation
  component of `module.transform`

Remove the `viz_mode`, `viz_scale`, `viz_rotation` rows if they exist.

## 4. Update `Canvas3D` rendering

**File:** `rayforge/ui_gtk/canvas3d/canvas3d.py`

In `_update_rotary_module_renderers()`:

- Remove the `CylinderRenderer` fallback entirely
- If `module.model_id` is `None`, skip — render nothing for that module
- If `model_id` is set, resolve via `model_mgr.resolve()` and create a
  `ModelRenderer`
- Apply `module.transform` when rendering:
  `mvp = scene_mvp @ module.transform`

The `CylinderRenderer` stays for workpiece visualization
(`_cylinder_renderers` keyed by `layer.rotary_diameter`).

## 5. Minimal downstream updates

- **`Layer`** (`rayforge/core/layer.py`): No changes. `rotary_module_uid`,
  `rotary_enabled`, `rotary_diameter` stay as-is.
- **`LayerSettingsDialog`** (`rayforge/ui_gtk/doceditor/layer_settings_dialog.py`):
  No changes.
- **`GcodeEncoder`** (`rayforge/pipeline/encoder/gcode.py`): Only reads
  `module.axis.name` — unaffected.
- **`DocEditor`** (`rayforge/doceditor/editor.py`): Only reads `module.uid`
  and `default_diameter` — unaffected.

## 6. Cleanup

- Delete `RotaryVisualizationMode` enum
- Remove parametric-specific test data from
  `tests/machine/models/test_rotary_module.py`
- Remove `rotary_visualization.md` (superseded by this document)
