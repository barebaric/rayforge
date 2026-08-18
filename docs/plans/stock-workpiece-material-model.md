# Unified Stock → Workpiece Material Model

Tracking PR for the architecture that unifies stock, workpiece, and
workpiece images for CNC and laser operations, and makes cut-through
operations punch real voids into the 3D stock preview.

## Tracking

- Iteration 0 — raygeo: effect types + fold kernel
- Iteration 1 — Pipeline integration (no visual change)
- Iteration 2 — Through-cut voids as real 3D geometry
- Iteration 3 — Laser raster cut-through detection
- Iteration 4 — Depth field (relief)
- Iteration 5 — Unified workpiece image
- Iteration 6 — Burn-in, rotary, simulation scrubbing
- Iteration 7 (future) — Solid profile: true 3D stocks and ops

This is revision 3 of the plan. It reworks the original proposal in
four ways:

- **Rust does the heavy lifting.** Effect extraction, folding,
  rasterization, contouring, mesh displacement, and image rendering
  all move into raygeo, executed on rayon worker threads with the GIL
  released — the same execution model the intent pipeline already
  uses for assemblers. Rayforge only builds plain-data specs,
  schedules work, and packs GPU buffers. No Python/numpy geometry
  fallbacks: where raygeo lacks a capability, we add it there (we own
  raygeo, per AGENTS.md).
- **No clipping to workpiece geometry.** The original plan derived
  workpiece results "clipped to the workpiece's world geometry".
  That is wrong: laser paths for a workpiece are routinely *larger*
  than the workpiece itself (frame steps add a rectangle around the
  part, overscan and lead-in/out transformers extend past contours,
  raster scan grids cover the whole workpiece bbox, and ops may cross
  stock edges). Effects live in world space, and the only geometric
  clip that is physically meaningful is against the *stock*: only
  material that exists can be removed.
- **Removal is volumetric; 2.5D is a fast path, not the model.**
  Revision 2 folded everything into 2D polygons plus a heightmap —
  which silently assumes vertical walls, top-accessible cuts, and
  prismatic stock. That breaks for genuinely 3D machining (sloped
  walls, undercuts, interior cavities, angled drilling into tube
  stock) and for future 3D stocks and workpieces. The model is now
  *profile-aware*: an exact, cheap **prismatic profile** (covers
  laser and all current CNC ops) and a **solid profile** (closed
  meshes, CSG subtraction) that future 3D assemblers and stocks drop
  into without redesign. See "Forward compatibility: the road to
  3D".
- **No phantom references.** Revision 1 cited a `material.md` plan
  (iterations C/D, P6.x, `_composite_burn_texture`,
  `cut_power_threshold` "from material.md"). No such document exists
  in the repository. Everything this plan needs — including the
  burn-into-stock visualization — is specified here.

## Problem statement

The current model has three disconnected representations of "the
material being worked on":

1. **Stock is static.** `StockItem` → `StockAsset` (geometry,
   thickness, `Material`). The 3D canvas renders it as an untouched
   prism (`stock_compiler.py` → raygeo `build_prism_mesh`) or a numpy
   rotary cylinder shell. Nothing an operation does ever changes it.
2. **Workpieces are image-based and created on demand.** A
   `WorkPiece` is a lightweight intent container (source segment /
   geometry provider + transform + tabs). Its "images" come from two
   ad-hoc paths: the pyvips render cache of the *source* (2D canvas)
   and `WorkPieceViewArtifact`, a bitmap of rendered *op strokes*
   (`pipeline/view/view_compute.py` → raygeo `render_ops`), neither
   of which represents the actual worked material.
3. **The raygeo `Part` is effectively the stock being operated on**
   during compute: `WorkPiece.to_part()`
   (`rayforge/core/workpiece.py:484`) builds the Part, assemblers
   mutate `Part.cleared` (`ClearedArea`, 2D swept-disk polygons), and
   `AssemblyOutput.cleared_fragments` carries the cut regions back —
   but rayforge currently uses them only for warnings and cache
   restore, then discards the outcome.

Laser and CNC also *diverge* in how they describe material effects:

|                      | CNC (milling)                 | Laser                           |
|----------------------|-------------------------------|---------------------------------|
| Removal tracking     | `ClearedArea` polygons + `target_z` | none                            |
| Surface effects      | none                          | R8 power maps (`rasterize_scanlines`) |
| 3D visualization     | none (flat stock)             | LUT power-texture quads floating over the stock; never burned in |
| Cut-through          | `target_z` ≤ stock bottom     | not detected, not visualized    |

Goal: one model in which every operation — CNC or laser — contributes
to an evolving representation of the stock, from which stock geometry,
workpiece geometry, and workpiece images are derived, and in which
cut-through operations punch actual voids into the 3D stock.

## Design principles

1. **Rust owns all geometry and pixel work.** Anything that touches
   polygons, grids, or meshes runs in raygeo on rayon threads. Python
   never iterates over vertices, polygons, or pixels on a hot path.
2. **Effects are world-space facts about the stock.** An effect is
   produced by an op of a workpiece, but its geometry may extend past
   the workpiece outline (frames, overscan, leads, raster bboxes) and
   past a single stock. Folding intersects effects with each stock;
   nothing is ever clipped to workpiece geometry.
3. **Effects describe removed volume.** The common language of CNC
   and laser is "this much material is gone, here". Prismatic
   effects express it as polygons extruded over a Z interval (CNC
   pockets, laser kerfs); raster effects as power maps with a
   power→depth response; future 3D assemblers as closed solids (tool
   swept volumes, relief surfaces). The fold subtracts volumes from
   the stock; how it represents the result is a per-stock choice of
   profile (§4), not a global assumption.
4. **2.5D is a fast path with explicit invariants, not the model.**
   The prismatic profile is valid exactly while: cuts are vertical
   (walls parallel to Z), every removed column is open to the top
   surface (no undercuts, no interior cavities), and the stock is a
   prism or a rotary wrap. The fold verifies these invariants and
   escalates to the solid profile when they break. Laser and every
   current CNC op satisfy them; nothing today pays 3D costs.

## Architecture: the Material Evolution Model

```
 (authoring)                (compute — all Rust, on rayon threads)
 WorkPiece ──to_part()──► Part ──assembler──► AssemblyOutput
 StockItem                                      │ ops, cleared_fragments,
 (world geometry)                               ▼ material_effects (new)
 └────────────► MaterialFold pipeline node ◄────┘
                   │  profile pick: prismatic (2.5D fast path)
                   │  or solid (CSG, for Volume effects/stocks)
                   ▼
            MaterialState (per stock, immutable)
            · prismatic: voids · depth_field · surface_map
            · solid: remaining-stock mesh + projections
              │                                          │
   raygeo build_stock_mesh                    raygeo render_material_view
   (prisms+relief / CSG result mesh)          (effect footprint + burn LUT)
              │                                          │
              ▼                                          ▼
       StockLayer GPU buffers                 Workpiece material image
```

### 1. `MaterialEffect` — per-op output, emitted in Rust

`AssemblyOutput` (a raygeo type) grows an optional field:

```rust
material_effects: Option<Vec<MaterialEffect>>
```

with `MaterialEffect` a Rust enum over *removed volume* descriptors:

- **`Vector { polygons, z_range }`** — exact 2D polygons extruded
  over a Z interval `[z_from, z_to]` (absolute, world Z after
  placement). The polygons are in workpiece-local mm; the fold places
  them into world space (§3). Sources: CNC assemblers (their swept/
  cleared geometry already exists in `FaceState.cleared` at assembly
  time; `z_from` = stock top, `z_to` = the spec's `target_z`; a
  groove that does not start at the surface carries its true
  interval); laser vector cut assemblers (the offset cut contour they
  already compute; a cut step spans the full stock thickness, an
  engrave vector step carries its focus depth / `z_step_down` per
  pass). MVP uses the offset contour as the void boundary;
  kerf-aware widening can refine it later. An interval — not a
  scalar bottom — so partial-depth and in-material prisms stay
  expressible.
- **`Raster { power, grid, response }`** — an R8 power map on a
  world-grid spec (origin_mm, px_per_mm), plus the material response
  used to interpret it (see §2). Sources: laser raster assemblers
  (scanlines are already in the assembled `Ops`; the fold reuses
  `raygeo.image.rasterize_scanlines`), CNC raster-style clearing
  where applicable. Raster effects are surface-domain (a depth per
  column from the exposed surface); both profiles consume them.
- **`Volume { solids }`** — closed manifold meshes (placed into world
  space by the fold, like vector polygons). No assembler emits these
  yet; the variant exists so that future 3D assemblers (3D relief
  machining, ball-nose surfacing, true swept-volume milling, angled
  drilling) and future non-prismatic stocks join the *same* fold
  without breaking `MaterialEffect`, `MaterialState`, or the
  pipeline wiring. A `Volume` effect escalates its stock to the
  solid profile (§3).

Emitters are the assemblers themselves, in Rust, where the shapes are
already in memory — *not* a Python-side conversion step in
`assembler_helpers.py` (revision 1's approach would have serialized
every fragment across the FFI and re-processed it under the GIL).
Assemblers that cannot emit yet simply leave the field `None` and the
fold derives an effect from their `Ops` + spec inside Rust instead;
the Python side never converts.

### 2. Material fabrication response — on `Material`, not appearance

`MaterialAppearance` (`rayforge/core/material.py:37`) is visual-only
(color, pattern, texture, PBR params); it stays that way. The new
fabrication parameters live on `Material` and are optional YAML
additions (absent ⇒ laser raster effects fold to surface effects
only, no cut-through detection):

- `cut_power_threshold: int | None` (0–255) — raster power at or
  above which the material is cut through.
- later, optional: `power_depth_curve` (piecewise power → removal
  depth) to make engraving depth physically plausible.

This is the single point where laser physics is translated into the
removal-volume language that CNC ops speak natively.

### 3. `MaterialFold` — a pipeline stage in raygeo

The fold runs *inside* the existing intent pipeline, not as a
Python-side pass:

- New stage spec `StageSpec.MaterialFold(material: MaterialFoldSpec)`
  alongside `Compute`/`Aggregate` — Python-visible constructor. On
  the Rust side this must respect raygeo's layering: `pipeline` is
  domain-free, so the fold node is a `pipeline::Compute`
  implementation living in the `cnc::execution` glue (exactly how
  `AssemblerCompute` wraps assemblers), not a new `StageSpec` enum
  variant. A fold node is keyed `stock:{stock_uid}` per visible
  stock item and depends on all `workpiece:{wp}:{step}` compute
  nodes whose workpiece world AABB intersects that stock. Stock world
  polygons come from the same `_resolve_stock_geometries` pass that
  already feeds `CropTransformer` (`pipeline/intent_builder.py`).
- The fold spec carries, as plain data: the stock shape (prism:
  polygons + thickness; rotary: wrap parameters; future: closed
  mesh), the per-source placement matrices (the same world
  transforms the aggregate stage uses to place workpiece-local ops
  into world/job space — compute outputs, including
  `cleared_fragments`, are workpiece-local), per-step fabrication
  params (Z ranges, through-cut flags, response threshold), and grid
  budget.
- `run_intent` already executes all nodes in a single `rayon::scope`
  with the GIL released for pure-Rust nodes; fold nodes join that
  pool, run in parallel with unaffected compute, and — on incremental
  rebuilds — read upstream `AssemblyOutput`s straight from the Rust
  cache when their compute nodes were cache hits. No Python round
  trip per effect.
- Caching and invalidation are the standard mechanism: the fold
  node's `version_token` hashes the stock revision plus all upstream
  tokens; unchanged stocks are cache hits.

**Profile selection.** The fold picks a representation per stock:

- **Prismatic** (the 2.5D fast path — laser and all current CNC):
  chosen when the stock is a prism or rotary wrap *and* every effect
  is `Vector`/`Raster` *and* the union of effect prisms is top-open
  (each removed column reaches the stock's top surface). Steps, all
  Rust, parallel where profitable:
  1. Collect upstream effects (cached or fresh), transform to world
     mm by the placement matrices.
  2. Through-cut classification: Z interval spans
     `[stock_top, stock_bottom]` ⇒ void.
  3. `void_polygons` = `get_polygons_union` of void effects, clipped
     to the stock by `get_polygons_group_difference` (Clipper2
     surface already exposed by raygeo).
  4. `depth_field`: stock-grid heightmap (AABB, 50 px/mm, capped at
     8192 px — today's power-texture budget), max-reduce of
     response-mapped raster powers and rasterized vector depths;
     rayon `par_iter` over tiles.
  5. `surface_map`: max-reduce of raw R8 powers (the burn input),
     composited per laser.
  6. Provenance (applied node keys) for epoch filtering and
     incremental re-folds.
- **Solid** (the general path): chosen when any effect is `Volume`,
  the stock is a non-prismatic closed mesh, or the prismatic
  invariants fail (undercuts, interior cavities, non-vertical walls).
  raygeo gains a solid-CSG capability for this (closed-manifold
  boolean subtraction — wrap a robust mesh-boolean crate such as
  `manifold`, or implement voxel/SDF booleans; raygeo already has
  planar 3D polygon booleans in `polygon3d`, which is *not*
  sufficient for solid CSG and is not used here). The fold converts
  prismatic effects to extruded solids and response-mapped rasters to
  displaced-surface solids, then subtracts sequentially from the
  stock solid on rayon (independent subtractions within one stock
  still parallelize per connected component).
- **Cross-profile derivation:** prismatic → solid is exact
  (extrusion); solid → prismatic views (voids, heightmap) is only
  valid without undercuts and is computed as a projection with a
  `approximated` flag when used (e.g., for a legacy renderer).

Escalation is decided per stock per fold, so one future 3D op never
degrades the performance of prismatic stocks in the same document.

### 4. `MaterialState` — immutable per-stock snapshot (Rust struct)

Result of the fold, tagged with its profile:

- `Prismatic`: world-space void polygons, `depth_field` and
  `surface_map` as `CompressedArray` + grid spec, provenance, and
  (iteration 6) marker-indexed cumulative snapshots for simulation
  scrubbing.
- `Solid`: the remaining-stock mesh as the f64 interchange solid
  (`geo::shape::solid::SolidMesh`), plus derived projections
  (silhouette/voids, top heightmap) where valid, provenance, and the
  same snapshot mechanism. Render meshes are derived from it at the
  `mesh::build` boundary, never stored in the state.

Returned to Python as plain data. Rayforge wraps it in a
`MaterialStateArtifact` (same store/generation-id infra as
`WorkPieceViewArtifact`; `rayforge/pipeline/artifact/material_state.py`
is a thin wrapper, not a compute module). Consumers ask for a
projection ("give me voids and a heightmap for this stock"), so
renderer code never assumes which profile produced the state.

### 5. Derived views — also Rust

**Evolved stock mesh.** raygeo's `mesh` module grows
`build_stock_mesh(state)`: for a prismatic state it subtracts voids
from the stock rings, displaces top-face vertices by the heightmap,
and recomputes normals — all inside Rust, building on the existing
`build_prism_mesh` ear-clip triangulation; for a solid state it
converts the remaining-stock solid (the CSG result is already
manifold). This is the boundary between the two mesh roles:
`MaterialState` carries the *interchange* mesh (f64 `SolidMesh`
from `geo::shape::solid`), and `mesh::build` produces the
*presentation* mesh — a `PrismMesh` with float32 positions,
computed normals, and UVs — which is what the public API returns
and what `stock_compiler.py` packs into GPU buffers, exactly as it
packs `build_prism_mesh` output today. The numpy rotary shell
builder (`stock_compiler._build_cylinder_shell`) moves to raygeo as
`build_cylinder_shell` with angular void gaps and radial
displacement. `stock_compiler.py` keeps exactly its documented role —
validate plain-data specs, pack GPU buffers — now passing material
grids/meshes through to Rust instead of building geometry itself.
There is deliberately no numpy displacement fallback.

**Workpiece material image.** A raygeo call
`render_material_view(effects, footprint, luts)` produces the RGBA
bitmap for the 2D canvas from the *effect footprint* — the union AABB
of that workpiece's own effects — intentionally **not** clipped to
the workpiece's source geometry, because the burned region can exceed
it. Depth-shaded + burn-LUT coloring per §2. `ViewManager` schedules
it exactly like today's op-stroke renders (same throttling and
concurrency caps); the op-stroke `WorkPieceViewArtifact` remains
available as a debug mode.

**Burn into stock.** `StockShader` gains a second texture sampler:
`MaterialState.surface_map` (R8) blended over the albedo through a
burn LUT — the same LUT machinery `ColorLutProvider` already provides
for `TextureShader`. The floating power-texture quads remain as the
playback trail; the burned stock is the persistent result. (This
specifies the capability revision 1 attributed to the nonexistent
`material.md`.)

### 6. raygeo `Part` role clarified (not replaced) — and its 3D future

The Part stays the compute-side stock proxy: assemblers keep using
`ClearedArea` for engagement/warnings inside Rust, and
`state_source_keys` still threads cleared state between CNC steps.
The bridge is at the output boundary: assembler effects feed the
fold. The difference to revision 1 is that the bridge is Rust→Rust.

This seam is also the 3D migration path: when `Part` grows an
optional solid stock shape and 3D assemblers subtract swept volumes
from a `ClearedVolume` (the 3D analogue of `ClearedArea`), they emit
`Volume` effects and the fold already knows what to do with them.
No redesign of `MaterialEffect`, `MaterialState`, the fold node, or
the renderer projections is needed at that point — that is the
property this revision buys.

### 7. What stays in Python

- `intent_builder`: emitting fold nodes, placement matrices, and
  fabrication params (dictionary work, microseconds).
- Artifact store put/get and signal plumbing.
- GPU buffer packing and GL upload via the existing chunked upload
  controller (already off-thread).

## Iterative implementation plan

### Iteration 0 — raygeo: effect types + fold kernel

- `MaterialEffect` (all three variants, including `Volume`, so the
  wire format never breaks), `MaterialState` with the profile tag,
  and a pure `fold_effects(spec) -> MaterialState` function (no
  pipeline integration yet): prismatic fold only — union/intersect,
  raster max-reduce, through-cut classification, the material
  response type (its depth application lands in iteration 4, raster
  void contouring in iteration 3), and the top-open invariant check
  with clean escalation signaling.
- `AssemblyOutput.material_effects` field (defaults `None`); CNC
  assemblers emit vector effects (polygon + Z interval); laser
  vector cut assemblers emit cut polygons with full-thickness Z
  intervals.
- Tests in raygeo: fold correctness (union, max-reduce, through-cut
  detection, placement transform), snapshot immutability, top-open
  invariant detection.

The detailed raygeo-side implementation and test plan for this
iteration follows below ("Phase 0 detail — raygeo implementation
and test plan"); it follows the raygeo `AGENTS.md` rules (layering,
explicit export paths, Python-only tests, generated stubs).

### Iteration 1 — pipeline integration (no visual change)

- raygeo: fold node as a `pipeline::Compute` implementation in the
  `cnc::execution` glue (the `AssemblerCompute` pattern; `StageSpec`
  itself stays domain-free), Python-visible as a
  `StageSpec.MaterialFold` constructor; dependency wiring,
  version-token caching, `rayon::scope` execution.
- rayforge: `intent_builder` emits `stock:{uid}` fold nodes with
  stock polygons, placements, and fabrication params; fold results
  wrapped into `MaterialStateArtifact`; raster steps derive effects
  in-fold from `Ops` via `rasterize_scanlines`.
- Tests: node keys/tokens, cache-hit refold (change one op ⇒ only
  its stock refolds), epoch filtering of stale folds.

### Iteration 2 — Through-cut voids as real 3D geometry

- raygeo: `build_stock_mesh` void subtraction (heightmap later);
  tests alongside `tests/mesh/test_mesh_prism.py`.
- rayforge: `stock_compiler` spec path + `RenderConfig3D` carry
  material state; `CompiledSceneArtifact`/`StockLayer` unchanged in
  shape (meshes just have holes); rebuilds ride the existing
  background compile.
- Tests: hole subtraction, mesh contains inner ring walls; manual:
  CNC profile cut and laser vector through-cut show real holes.

### Iteration 3 — Laser raster cut-through detection

- `Material.cut_power_threshold` YAML field (additive, optional).
- raygeo: threshold the raster power map ⇒ cut mask ⇒ contours
  (marching squares, new in raygeo — no Python fallback) ⇒
  through-cut polygons into `void_polygons`, unioned with vector cut
  voids.
- Tests: threshold contouring, union with vector cuts, absent
  threshold ⇒ no voids.

### Iteration 4 — Depth field (relief)

- raygeo: `depth_field` accumulation (CNC: `stock_top − target_z`;
  raster: response-mapped depth) and `build_stock_mesh` heightmap
  displacement + normal recompute; `build_cylinder_shell` ported from
  numpy with radial displacement.
- rayforge: pass grids through the stock spec; delete the numpy shell
  builder.
- Tests: displacement values, normals; manual: pockets and deep
  engraving visibly carved into the stock.

### Iteration 5 — Unified workpiece image

- raygeo: `render_material_view(effects, footprint, luts)`.
- rayforge: `ViewManager` renders it from the workpiece's effect
  footprint (explicitly never clipped to workpiece geometry);
  consumers migrate gradually, op-stroke view stays as debug mode.

### Iteration 6 — Burn-in, rotary, simulation scrubbing

- `StockShader` burn sampler consuming `surface_map` (new capability,
  specified in §5; nothing pre-existing is replaced).
- Rotary: angular voids, shell relief, cylinder burn wrap.
- Simulation scrubbing: fold emits marker-indexed cumulative
  snapshots; the `OpPlayer` maps playback time to markers, making
  "material removal over time" incremental over prefix unions.

### Iteration 7 (future) — Solid profile: true 3D stocks and ops

Not scheduled with the iterations above; it lands when 3D
assembler work starts. Nothing in iterations 0–6 needs rework:

- raygeo: solid-CSG module (closed-manifold booleans — e.g. wrapping
  a crate like `manifold`, or voxel/SDF booleans; CSG-internal mesh
  types convert to `geo::shape::solid::SolidMesh` at the fold
  boundary); solid fold path (prismatic→solid conversion, sequential
  subtraction, projections); `build_stock_mesh` solid→`PrismMesh`
  conversion; tests alongside new raygeo mesh tests.
- raygeo `Part`: optional solid stock shape; `ClearedVolume`
  engagement state for 3D assemblers; first 3D assembler emits
  `Volume` effects.
- rayforge: `StockAsset` gains an optional closed-mesh geometry
  alongside `geometry`/`thickness`; `StockItem.get_world_geometry`
  analogue for solids; scene compiler consumes solid states.
- Tests: CSG correctness in raygeo; rayforge wiring only.

## Forward compatibility: the road to 3D

What the prismatic profile assumes, and what breaks it:

| Prismatic invariant           | Broken by                            |
|-------------------------------|--------------------------------------|
| Walls parallel to Z           | sloped/3D-surface machining, angled drilling |
| Every removed column open to the top | undercuts, interior cavities, grooves with material above |
| Stock is a prism or rotary wrap | square tubing, extrusions with pockets, cast/molded blanks |

The architecture stays valid across all of it because the
assumptions live in exactly two places — profile selection in the
fold (§3) and the projection helpers on `MaterialState` (§4) — and
every consumer (stock meshing, workpiece image, burn-in, scrubbing)
goes through those projections. Adding the solid profile is additive
Rust work in raygeo; the Python pipeline, artifact plumbing, and
renderers are untouched except for passing meshes through.

## Performance

- Fold nodes execute in the existing `run_intent` `rayon::scope`,
  GIL released, parallel with compute and with each other (per
  stock). Upstream cache hits feed the fold without Python.
- Prismatic fold stays 2D (Clipper2 + tiled grids): today's cost
  profile is unchanged by the volumetric model. Grid work is tiled
  and `par_iter`-ed; budgets match today's power textures
  (50 px/mm, ≤ 8192 px).
- Solid-profile CSG costs are paid only by stocks that escalated;
  prismatic stocks in the same document are unaffected. CSG work is
  bounded by effect count and mesh complexity, and the fold caches
  per stock like any other node.
- Mesh rebuilds run in Rust on the existing scene-compile worker
  thread, rayon-parallel across stock layers.
- Nothing on any hot path holds the GIL or round-trips geometry
  through Python.

## Compatibility & risks

- `material_effects` defaults to `None`: old assemblers and old
  caches remain valid; new rayforge requires a raygeo version bump
  (we own both).
- Material YAML additions are optional; absent fields mean fewer
  derived effects, never errors.
- Absent a `MaterialState`, stock and workpiece rendering behave
  exactly as today (specs without material state).
- For iterations 0–6 no 3D CSG is needed: voids are 2D polygon
  booleans plus the existing hole-aware prism build; relief is vertex
  displacement. The solid profile (iteration 7) introduces a CSG
  dependency in raygeo — the main new risk is boolean robustness;
  mitigated by using a proven manifold-boolean crate, keeping CSG
  behind the fold boundary, and falling back to prismatic
  projections for consumers while a solid stock's state computes.
- Top-open checking adds a cheap 2D pass to the prismatic fold
  (per-column reachability against the depth field); failure is an
  escalation, not an error.
- Multi-stock documents fold per stock by intersection; a workpiece
  spanning two stocks contributes to both. Workpiece images are
  stock-independent.

## Testing

- Geometry and fold precision tests live in raygeo (upstream repo),
  following its existing `tests/mesh/` patterns: fold correctness per
  profile, profile selection and escalation (top-open violations,
  `Volume` effects, non-prismatic stocks), cross-profile conversions.
- rayforge tests cover wiring, not geometry: intent-builder node
  emission (pattern: `tests/pipeline/`), artifact round-trips
  (pattern: `tests/simulator/test_compiled_scene.py`), stock compiler
  buffer packing (pattern: `tests/simulator/test_stock_compiler.py`),
  presenter scheduling (pattern: `tests/ui_gtk/sim3d/test_scene_presenter.py`).

## Dependency ordering

```
Iteration 0 ──► 1 ──► 2 (vector voids) ──► 3 (laser voids) ──► 6
                 └───► 4 (relief) ──► 5 (unified image) ──► 6

Iteration 7 (solid profile) builds on 0's types and 6's snapshot
infra; independent of iterations 2–5.
```

Iterations 2 and 4 are parallelizable after 1; 3 depends on 2; 5
depends on 4; 6 integrates everything. Iteration 7 is future work
and blocked only by 3D assembler capability, not by this plan's
iterations.

## Phase 0 detail — raygeo implementation and test plan

Everything in this section happens in the raygeo repository and
follows its `AGENTS.md`: layering (`geo → ops → cnc`, `pipeline`
domain-free, never import upward), explicit export paths (no
re-exports in parent `mod.rs`), stubs only via `make stubs`, docs
only via `make docs`, and **no Rust tests** — all tests are Python
under `tests/` exercising the code through PyO3 bindings.

### Module layout and layering

Material-effect folding classifies removal — an ops-layer concern
("what is cut"), built on geo primitives. It must be reachable from
`ops::assembly` (the `AssemblyOutput` field) without upward imports,
so the core lives at `src/ops/material/`:

```
src/ops/material/
├── mod.rs     MaterialEffect, FoldProfile, Escalation
│              (SolidMesh itself lives in geo, see below)
├── spec.rs    MaterialFoldSpec, FoldEntry, StockShape, GridBudget,
│              MaterialResponse
├── state.rs   MaterialState
├── fold.rs    fold_effects(&MaterialFoldSpec) -> MaterialState
│              (pure Rust, no PyO3, rayon where profitable)
└── grid.rs    raster resampling / max-reduce helpers
```

- `pub mod material;` in `src/ops/mod.rs` — nothing else; items are
  imported by their leaf path (`crate::ops::material::...`).
- `fold.rs` uses `crate::geo::shape::polygon::{get_polygons_union,
  get_polygons_group_difference}` (Clipper2 wrappers, XY-plane) and
  `crate::compressed_array::CompressedArray` for grids — both
  downward/sideways-safe imports. `rayon` is already a dependency.
- Bindings: one new leaf `src/python/ops/material.rs` following the
  `material_test_grid` precedent: `pub(crate) fn register(m)`,
  submodule added in `src/python/ops/mod.rs::register` plus
  `sys_modules.set_item("raygeo.ops.material", &m)` so imports
  resolve; `#[pyclass(module = "raygeo.ops.material")]`,
  `#[gen_stub_pyclass]`/`#[gen_stub_pymethods]`/
  `#[gen_stub_pyfunction(module = "raygeo.ops.material")]`, and the
  `pyo3_stub_gen::module_doc!` macro. The `fold_effects` binding
  releases the GIL via `py.detach(...)` while the kernel runs (same
  pattern as `python/image/rasterize_scanlines.rs`).
- `Volume { solids }` carries a plain-data `SolidMesh { positions:
  Vec<Point3D>, triangles: Vec<[u32; 3]> }` (f64, closed manifold).
  It is *not* internal wiring — it is the public interchange format:
  it crosses the cache boundary inside `AssemblyOutput`, the PyO3
  boundary as `raygeo.ops.material` objects, and (iteration 7) back
  to consumers inside `MaterialState::Solid`. Declining the `mesh`
  module's types is therefore not about hiding internals — it is
  that both are presentation/domain formats of the wrong shape:
  `PrismMesh` is float32 GPU buffers with normals and UVs baked in,
  `TriangleMesh` is a planar 2D FEM mesh with adjacency and boundary
  tags. Interchange wants f64 positions plus triangle indices and
  nothing else; sharing render types would also make every
  render-side format change invalidate the ops cache format. Since
  both `ops::material` and `mesh` need it, it lives one layer down
  as a pure geometric primitive: new leaf `geo/shape/solid.rs`
  (both import downward; no `ops → mesh` edge is created).
- After changing bindings: `make stubs` (regenerates
  `python/raygeo/**/__init__.pyi`; never hand-edit), `make docs`
  (markdown docs are generated). Both artifacts are committed.

### Core types (Rust)

```rust
pub enum MaterialEffect {
    Vector {
        polygons: Vec<Polygon>,      // workpiece-local mm, XY
        z_from: Option<f64>,         // None = open to stock top
        z_to: Option<f64>,           // None = through stock bottom
    },
    Raster {
        power: CompressedArray,      // R8, shape [rows, cols]
        grid: GridSpec,              // origin_mm + px_per_mm
        response: MaterialResponse,
    },
    Volume {
        solids: Vec<SolidMesh>,      // world-space after placement
    },
}

pub struct FoldEntry {
    pub source_key: String,          // node key, for provenance
    pub placement: Matrix,           // XY placement into world mm
    pub effects: Vec<MaterialEffect>,
}

pub enum StockShape {
    Prismatic { polygons: Vec<Polygon>, thickness: f64 },
    // Rotary and Solid variants arrive with iterations 4 and 7;
    // phase 0 rejects them with a clear error, not a panic.
}

pub struct MaterialFoldSpec {
    pub stock: StockShape,
    pub entries: Vec<FoldEntry>,
    pub grid: GridBudget,            // px_per_mm (default 50), max_px
}                                   // per side (default 8192)
```

`z_from`/`z_to` as `Option<f64>` is the seam that keeps the model
volumetric: laser through-cuts are `Vector { z_from: None, z_to:
None }`; adaptive/profile pockets are `{ z_from: None, z_to:
Some(target_z) }`; a future groove starting below the surface sets
`z_from: Some(z)` — and that is exactly the case the top-open check
flags. Absolute world Z; placement matrices only move XY (matches
the aggregate stage today).

`MaterialState` (phase-0 fields; `depth_field` stays `None` until
iteration 4):

```rust
pub struct MaterialState {
    pub profile: FoldProfile,        // Prismatic in phase 0
    pub void_polygons: Vec<Polygon>, // world mm, clipped to stock
    pub depth_field: Option<CompressedArray>,  // f32 mm, iter. 4
    pub surface_map: Option<CompressedArray>,  // R8 max power
    pub grid: Option<GridSpec>,
    pub provenance: Vec<String>,     // sorted source keys
    pub escalation: Option<Escalation>, // e.g. TopOpenViolation
}
```

### Fold kernel behavior

`fold_effects` is deterministic and order-independent (union and
max are commutative, associative; provenance is sorted by source
key), so cache tokens and snapshot equality are stable. Steps:

1. Transform each entry's effect polygons to world mm by
   `placement` (raster grids: adjust `grid.origin_mm`; if the
   placement scales, record it and resample in step 4).
2. Validate: reject unknown `StockShape` variants; clamp grid
   budget; `MaterialResponse` threshold in `0..=255`.
3. Through-cut classification (vector effects only in phase 0):
   `z_to` is `None` or `<= stock_bottom`, and `z_from` is `None` or
   `>= stock_top` ⇒ void candidate. A `Some(z_from) < stock_top`
   start below the surface sets `Escalation::TopOpenViolation`
   (phase 0 reports; the solid profile that resolves it arrives in
   iteration 7 — the signal, not a panic).
4. `void_polygons` = `get_polygons_union(void candidates)`, then
   clipped to the stock by `get_polygons_group_difference` (only
   material that exists can be removed).
5. `surface_map`: resample every raster effect's R8 grid onto the
   stock AABB grid (`grid.rs`: nearest-pixel mapping with
   max-reduce, `rayon::par_iter` over row tiles), compositing per
   pixel the maximum power. Empty result ⇒ `surface_map: None`.
6. Provenance: sorted unique `source_key`s.

Grid resolution: stock AABB at `px_per_mm`, each side capped at
`max_px` (rescale `px_per_mm` down to fit) — the same budget as
today's power textures.

### Emitters — `AssemblyOutput` integration

- `AssemblyOutput` (src/ops/assembly/mod.rs) grows
  `material_effects: Option<Vec<MaterialEffect>>`, default `None`.
- Mechanism: `AssembleCtx` gains an `effects: Vec<MaterialEffect>`
  out-box, drained by `AssemblerCompute` per face alongside the
  tracelet — the exact pattern `Ops` emission already uses, so no
  `Assembler` trait change and every assembler can emit.
- Phase-0 emitters (each pushes once per face at the end of
  `assemble`):
  - adaptive clearing and profile inner/outer:
    `Vector { polygons: ctx.face.cleared.fragments().to_vec(),
    z_from: None, z_to: Some(spec.target_z) }`. Effects derived from
    face state may re-describe upstream cuts after
    `state_source_keys` threading — harmless, fold union is
    idempotent; delta-only emission can be refined later.
  - laser contour: `Vector { polygons: <the offset cut contour>,
    z_from: None, z_to: None }` — the assembler already computes the
    kerf-offset geometry (`grow_geometry` in `assemble_contour`).
  - All other assemblers: no emission (`None`) — raster derives
    in-fold from `Ops` starting in iteration 1.
- Cache path updates (all in `src/cnc/execution/compute.rs`):
  `prepare_cache_entry` size estimate adds effect polygon vertex
  counts; the adaptive assembler's `store_cache` override rebuilds
  `AssemblyOutput` field-by-field and **must forward the new field**
  (easy to miss — its constructor enumerates every field); the
  Python `PyAssemblyOutput` and the `any_to_py` converter expose
  `material_effects` as a list of `raygeo.ops.material` objects.

### Python API (test-facing surface)

```python
from raygeo.ops.material import (
    FoldEntry, GridBudget, MaterialFoldSpec, MaterialResponse,
    PrismaticStock, RasterEffect, VectorEffect, fold_effects,
)

spec = MaterialFoldSpec(
    stock=PrismaticStock(polygons=[stock_ring], thickness=3.0),
    entries=[FoldEntry(
        source_key="workpiece:w1:step1",
        placement=placement_matrix,
        effects=[VectorEffect(polygons=[cut_ring],
                              z_from=None, z_to=None)],
    )],
    grid=GridBudget(px_per_mm=50.0, max_px=8192),
)
state = fold_effects(spec)   # GIL released during fold
state.profile, state.void_polygons, state.surface_map, state.grid
state.provenance, state.escalation, state.depth_field
```

PyClasses convert to core types via an `into_core()` pattern (same
as the profile spec binding); `MaterialEffect` itself stays a Rust
enum, constructed from the three concrete effect classes.

### Test plan (all Python, no Rust tests)

New `tests/ops/material/` mirrors the source tree; emitter tests
that drive the pipeline live in `tests/cnc/execution/` to reuse its
`conftest.py` builders (`make_square_part`, `make_contour_compute`,
`collect_completions`, autouse cache clearing).

`tests/ops/material/test_fold_vector.py` — helpers `_rect`/`_circle`
per file, matching existing assembler tests:

- through-cut (`z_to=None`; `z_to <= stock_bottom`) ⇒ void polygon
  present; partial depth (`z_to > stock_bottom`) ⇒ absent
- void clipped to stock: effect polygon straddling the stock edge
  comes back intersected; effect fully outside ⇒ no voids
- overlapping voids from two entries ⇒ single unioned polygon (area
  via `get_polygon_area`, `pytest.approx`)
- placement: translated/rotated/scaled `FoldEntry.placement`
  transforms effect polygons (assert on vertex positions)
- escalation: `z_from` below the stock surface ⇒
  `state.escalation == TopOpenViolation`, fold still completes
- multi-entry provenance sorted, duplicates collapsed

`tests/ops/material/test_fold_raster.py`:

- max-reduce: two overlapping rasters ⇒ per-pixel maximum
- resampling onto stock grid: different `origin_mm`/`px_per_mm`
  land at the right world coordinates (line-shaped raster ⇒
  expected row/col of nonzero pixels)
- grid cap: `px_per_mm` rescaled when AABB would exceed `max_px`
- no raster entries ⇒ `surface_map is None`
- `MaterialResponse` validation (threshold bounds, curve piecewise
  monotonic once added) rejects bad values with `ValueError`

`tests/ops/material/test_fold_state.py`:

- immutability: mutate the spec's effect polygons after folding ⇒
  folded `state` unchanged; fold twice ⇒ equal voids/surface maps
- `StockShape` variants not yet supported raise a clear `ValueError`
- `Volume` effects present ⇒ `Escalation` set (phase-0 semantics),
  no crash

`tests/cnc/execution/test_material_effects_emission.py`:

- adaptive clearing compute node ⇒ `AssemblyOutput.material_effects`
  is a non-empty list of vector effects with `z_to == target_z` and
  polygons matching `cleared_fragments` (same construction pattern
  as `test_ops_assembly_adaptive.py`, driven through `execute_stages`
  like the conftest tests)
- contour (laser) compute node ⇒ one full-through vector effect
  (`z_from`/`z_to` both `None`) per face
- re-running the same node (cache hit) ⇒ effects restored intact
  (guards the `store_cache`/`restore_cache` forwarding)
- multi-face part ⇒ effects concatenated across faces

Slow marker: one `@pytest.mark.slow` case folding a large synthetic
raster (e.g. 4096²) to keep the max-reduce path honest without
slowing every run.

### Workflow and definition of done

Per raygeo `AGENTS.md`, using make targets only:

1. `make dev` — build and install into the venv (required before
   testing).
2. Targeted tests while iterating:
   `pytest tests/ops/material -v` and
   `pytest tests/cnc/execution/test_material_effects_emission.py -v`.
3. `make format` — `cargo fmt` + ruff format/fix.
4. `make lint` — `cargo fmt --check`, `cargo clippy -- -D warnings`,
   ruff, and pyright over `python/raygeo tests tools` (CI runs the
   same and validates wheel stubs).
5. `make stubs`, `make docs` — regenerate and commit `.pyi` and docs
   after binding changes.
6. `make check` (lint + `cargo test` + `pytest -v`) green, with no
   new Rust `#[cfg(test)]` blocks added.

Deliverables: the `ops::material` module + bindings + regenerated
stubs/docs, the `AssemblyOutput.material_effects` field with
emitters for adaptive/profile/contour, and the test files above —
no rayforge changes in phase 0.
