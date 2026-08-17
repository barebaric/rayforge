# Unified Stock → Workpiece Material Model

Tracking PR for the architecture that unifies stock, workpiece, and
workpiece images for CNC and laser operations, and makes cut-through
laser operations produce real voids in the 3D stock preview.

## Problem statement

The current model has three disconnected representations of "the
material being worked on":

1. **Stock is static.** `StockItem` → `StockAsset` (geometry,
   thickness, `Material`). The 3D canvas renders it as an untouched
   prism (`stock_compiler.py` → `build_prism_mesh`) or rotary cylinder
   shell. Nothing an operation does ever changes it.
2. **Workpieces are image-based and created on demand.** A
   `WorkPiece` is a lightweight intent container (source segment /
   geometry provider + transform + tabs). Its "images" come from two
   ad-hoc paths: the pyvips render cache of the *source* (2D canvas)
   and `WorkPieceViewArtifact`, a bitmap of rendered *op strokes*
   (`pipeline/view/view_compute.py`), neither of which represents the
   actual worked material.
3. **The raygeo `Part` is effectively the stock being operated on**
   during compute: `WorkPiece.to_part()`
   (`rayforge/core/workpiece.py:484`) builds the Part, assemblers
   mutate `Part.cleared` (`ClearedArea`, 2D swept-disk polygons), and
   `AssemblyOutput.cleared_fragments` carries the cut regions back —
   but rayforge currently uses them only for warnings and cache
   restore, then discards the outcome.

Laser and CNC also *diverge* in how they describe material effects:

| | CNC (milling) | Laser |
|---|---|---|
| Removal tracking | `ClearedArea` 2D polygons + `target_z` | none |
| Surface effects | none | R8 power maps (`rasterize_scanlines`) |
| 3D visualization | none (flat stock) | LUT quads; GPU burn-in planned (`material.md` iterations C/D) |
| Cut-through | conceptually `target_z` ≤ stock bottom | deferred (`material.md` Phase 6, shader-discard MVP) |

Goal: one model in which every operation — CNC or laser — contributes
to an evolving representation of the stock, from which workpiece
geometry *and* workpiece images are derived, and in which cut-through
operations punch actual voids into the 3D stock.

## Architecture: the Material Evolution Model

```
                    (authoring, unchanged)                (compute)
  WorkPiece ──to_part()──► raygeo Part ──assemblers──► Ops + outputs
     │                                                     │
     │                                             MaterialEffect(s)
     │                                             (unified, per op)
     │                                                     │ fold
     │                                            ▼
     │                                     MaterialState (per stock,
     │                                       immutable snapshot)
     │                                   ┌─────┴─────┬──────────┐
     │                              void_polygons  depth_field  surface_map
     │                                   │             │            │
     └────────── derived views ◄─────────┴─────────────┴────────────┘
        WorkpieceResult (3D spec + unified 2D image), clipped to
        the workpiece's world geometry
```

### 1. `MaterialEffect` — unified per-op output

Every operation, regardless of family, emits effects in one of two
forms (or both):

- **Vector effect**: exact 2D `polygons` + absolute Z range
  `[z_from, z_to]`. Sources: CNC contour/profile/adaptive assemblers
  (from `cleared_fragments` + `target_z`), laser vector cuts (cut
  polygons with full-thickness range for through-cuts, or focus-depth
  range for engraving).
- **Raster effect**: `power_texture` (R8, stock-grid px/mm) + a
  *material response* mapping power → removal depth. Sources: laser
  raster steps (`rasterize_scanlines` output already exists), CNC
  raster-style clearing where applicable.

The material response lives on `MaterialAppearance`
(`cut_power_threshold` from `material.md` P6.1, optionally a
power→depth curve later). This is the single point where laser
physics is translated into the same "remove material down to depth
Z" language that CNC ops speak natively.

### 2. `MaterialState` — immutable per-stock snapshot

Folded by the pipeline in document/operation order, cached as a
pipeline artifact (same store/generation-id infra as
`WorkPieceViewArtifact`):

- `void_polygons`: 2D regions removed through the full stock
  thickness (union of effects whose Z range spans `[stock_top,
  stock_bottom]`).
- `depth_field`: single-channel heightmap in mm of removal depth on
  the stock's world grid (max-reduce of raster depths and
  rasterized/vector polygon depths).
- `surface_map`: R8 burn/power composite — exactly what
  `material.md` iteration C.1 already builds per stock layer; it
  becomes an output of `MaterialState` instead of a renderer-side
  composite.
- provenance (applied op keys) for incremental re-folds and cache
  invalidation.

Folding uses raygeo's existing Clipper2 surface: `get_polygons_union`,
`get_polygons_group_difference`, and `ClearedArea` semantics
(`remaining()` = stock minus cuts) as reference.

### 3. Workpiece as a derived view

`WorkPiece` keeps its authoring role (source, transform, tabs,
`to_part()` for compute). The *result* becomes a new derived
artifact:

- **`WorkpieceResult`** — computed from the final `MaterialState`
  clipped to the workpiece's world geometry
  (`get_world_geometry`-style transform). Provides:
  - a 3D geometry spec (outline with voids, relief heights, burn
    texture reference) for the scene compiler, and
  - the **unified workpiece image**: a 2D bitmap rendered from
    `depth_field` + `surface_map` (depth-shaded + burn LUT) replacing
    the ops-stroke `WorkPieceViewArtifact` rendering over time.

  "Created on demand" is preserved — but demand is now "MaterialState
  changed," driven by the existing pipeline cache, not by UI-side
  image rendering of strokes.

### 4. raygeo `Part` role clarified (not replaced)

The Part stays the compute-side stock proxy: assemblers need
`ClearedArea` for engagement/warnings inside Rust. The bridge is at
the output boundary: assembler results (`cleared_fragments`,
`target_z`, scanline power maps) are converted to `MaterialEffect`s
and folded. No assembler changes required.

### 5. 3D rendering of evolved stock

- **Voids — real geometry holes** (supersedes the `material.md`
  P6.2(a) shader-discard MVP; keep discard as a fallback for huge cut
  counts): `stock_compiler.py` subtracts `void_polygons` from the
  stock outline via `get_polygons_group_difference` and passes them
  as `holes` to `build_prism_mesh`, which already carves top-face
  holes and builds inner ring walls. Rotary stocks: voids become
  angular gaps in the cylinder shell.
- **Relief**: prism top-face vertices displaced by `depth_field`
  (raygeo extension to `build_prism_mesh`: heightmap sampling /
  per-vertex displacement; fallback: numpy displacement + normal
  recompute in `stock_compiler`). Rotary: shell radius displacement.
- **Burn**: the StockShader power-texture path planned in
  `material.md` iterations C/D, now consuming
  `MaterialState.surface_map` instead of a renderer-side composite.

## Iterative implementation plan

### Iteration 1 — `MaterialEffect` plumbing (no visual change)

- New `rayforge/pipeline/artifact/material_state.py`:
  `MaterialEffect`, `MaterialState`, fold functions
  (`fold_vector_effect`, `fold_raster_effect`, through-cut
  classification against stock thickness).
- Emit effects where results already exist:
  `pipeline/stage/assembler_helpers.py` converts
  `AssemblyOutput.cleared_fragments` + step `target_z` into vector
  effects (CNC); laser vector-cut steps emit polygon effects; raster
  steps emit power maps (no depth yet).
- Tests: fold correctness (union, max-reduce, through-cut
  detection), snapshot immutability, provenance-based incremental
  re-fold.

### Iteration 2 — Through-cut voids as real 3D geometry

- `stock_compiler.py`: accept `MaterialState` on the spec; subtract
  `void_polygons` from `outers`, pass as `holes` to
  `build_prism_mesh`; rebuild triggers via existing background
  compile.
- `scene_presenter._build_stock_specs` and `CompiledSceneArtifact`
  carry void polygons.
- `VisibilityOverlay` "Show cut-through" toggle (re-scoped
  `material.md` P6.3).
- Tests: hole subtraction, mesh contains inner walls; manual:
  CNC profile cut + laser vector through-cut show real holes.

### Iteration 3 — Laser cut-through detection

- `MaterialAppearance.cut_power_threshold` (P6.1) drives the raster
  fold: threshold the power map → cut mask → contours (marching
  squares, added to raygeo `image` or Python-side) → through-cut
  polygons into `void_polygons`.
- Vector laser cuts with full-thickness Z range feed iteration 2
  directly.
- Tests: threshold contouring, void union with vector cuts.

### Iteration 4 — Depth field (relief)

- Depth accumulation: CNC effects contribute
  `stock_top − target_z`; raster effects contribute response-mapped
  depth. Stored as `depth_field` on `MaterialState`.
- raygeo: `build_prism_mesh` heightmap/displacement support (we own
  raygeo — fix/extend there, per AGENTS.md); numpy fallback in
  `stock_compiler` + normal recompute.
- Tests: displacement values, normals; manual: pockets and deep
  engraving visibly carved into the stock.

### Iteration 5 — Unified workpiece image

- `WorkpieceResult` artifact: 2D bitmap from `depth_field` +
  `surface_map` clipped to workpiece geometry; migrate
  `WorkPieceViewArtifact` consumers gradually (ops-stroke view
  remains available as a debug mode, like `show_lut_overlay`).

### Iteration 6 — Burn-in integration + polish

- Wire `surface_map` as the input of the `material.md` C/D burn
  composite (replaces renderer-side `_composite_burn_texture`).
- Rotary: angular voids, shell relief, D.1 burn wrap.
- Simulation playback scrubbing: op-by-op `MaterialState` snapshots
  make "material removal over time" a freebie.

## Compatibility & risks

- Fully additive: absent a `MaterialState`, stock/workpiece rendering
  behaves exactly as today; `material.md` C/D burn-in plan is
  unchanged except for where the composite is computed.
- Mesh rebuild cost for voids/relief stays on the existing background
  compile thread; Clipper2 handles arbitrary cut shapes.
- raygeo has no 3D CSG — none is needed: voids are 2D polygon
  booleans + hole-aware prism build; relief is vertex displacement.
- `depth_field` resolution is bounded by the stock AABB at 50 px/mm
  (capped), same budget as today's power textures.

## Dependency ordering

```
Iteration 1 ──► 2 (vector voids) ──► 3 (laser voids) ──► 6
          └────► 4 (relief) ──► 5 (unified image) ──► 6
```

Iterations 2 and 4 are parallelizable after 1; 3 depends on 2; 5
depends on 4; 6 integrates everything.
