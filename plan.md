Here is the comprehensive refactoring plan for the Rayforge Artifact Management system.

## Project Objective
**Transition from a distributed, reactive pipeline to a centralized, intent-driven architecture.**

Currently, pipeline stages independently decide what to build and how to name artifacts, leading to synchronization issues and memory leaks. The new architecture creates a strict separation of concerns:
1.  **The Pipeline** determines *what* needs to be built (The Intent) and assigns a Global Version (Generation ID).
2.  **The Manager** acts as the strict ledger of state, tracking generic keys against Generation IDs.
3.  **The Stages** become stateless executors that simply fulfill the work defined by the Pipeline.

---

## Core Concepts

*   **GlobalGenerationID**: An integer representing a consistent snapshot of the document state. Incremented by the Pipeline on every document change.
*   **ArtifactKey**: A generic, type-safe identifier (UUID + Group) representing a logical entity (e.g., "Step 1", "Workpiece A"). It identifies the *entity*, while the GenerationID identifies the *version*.
*   **Intent**: The set of all ArtifactKeys that *should* exist for a specific GenerationID.

---

## The 7-Step Refactoring Plan

### Step 1: Introduce Generic `ArtifactKey` & Global ID
**Objective:** Eliminate ad-hoc tuple keys (e.g., `("workpiece", step_uid, wp_uid)`) and stage-specific ID counters.

**Implementation:**
1.  Create a generic `ArtifactKey` data structure containing:
    *   `id`: A UUID unique to the logical entity (e.g., the Workpiece).
    *   `group`: A string for categorization (e.g., "workpiece", "step", "job").
2.  Update the `Pipeline` to maintain a mapping of `DocItem` -> `ArtifactKey`. When a DocItem is created, it gets a stable `ArtifactKey`.
3.  Add `_global_generation_id` to the `Pipeline`.
4.  Update `ArtifactManager` to accept `ArtifactKey` in all its APIs instead of tuples.
5.  Refactor Stages to request keys from the Pipeline (or construct them via helper) using the new structure.

**Verification:** The application functions identically, but all dictionary keys in the Manager are now uniform `ArtifactKey` objects.

### Step 2: Enforce Strict State Contracts in Manager
**Objective:** Prevent ambiguous state updates by renaming APIs to reflect specific lifecycle events.

**Implementation:**
1.  Rename and restrict `mark_pending` to `schedule_for_generation(key, gen_id)`.
    *   **Logic:** Must create a NEW entry. Fails if entry already exists. Sets state to `PENDING`.
2.  Rename and restrict `commit` to `commit_artifact(key, handle, gen_id)`.
    *   **Logic:** Must find an exact match for `(Key, GenID)` in `PENDING` state. Transitions to `READY`.
3.  Rename `mark_error` to `fail_generation(key, error, gen_id)`.
4.  Update all Stages to use these explicit methods.

**Verification:** Any "ghost" tasks from previous generations trying to commit results will now raise exceptions (visible in logs) instead of silently corrupting state.

### Step 3: Purge State from Stages (Pass-through ID)
**Objective:** Make stages stateless workers that operate solely on the context provided for a specific execution cycle.

**Implementation:**
1.  Remove all internal `_current_generation_id` or `_next_generation_id` counters from Stage classes.
2.  Update `Stage.reconcile(doc, generation_id)` to accept the Global ID from the Pipeline.
3.  Ensure this `generation_id` is passed deeply:
    *   From Stage -> Background Runner (subprocess).
    *   From Runner -> Event (IPC).
    *   From Event Handler -> Manager `commit_artifact`.
4.  Stages must treat the `generation_id` as an opaque token they are required to stamp on all their outputs.

**Verification:** Trigger rapid document changes. Confirm via logs that runners are processing different IDs and correctly reporting them back, with no cross-talk between generations.

### Step 4: Centralize Intent (The "Declare" Phase)
**Objective:** Shift the responsibility of "what needs to exist" from distributed Stages to the central Pipeline.

**Implementation:**
1.  Add `manager.declare_generation(gen_id, keys)` API.
    *   **Logic:** Creates `MISSING` entries for all provided keys for this ID.
2.  Refactor `Pipeline.reconcile_all`:
    *   Traverse the Document model.
    *   Collect `ArtifactKeys` for every item that needs an artifact (Job, Steps, Workpieces).
    *   Call `manager.declare_generation` with this full list.
3.  Refactor `Stage.reconcile`:
    *   Stop traversing the document to find work.
    *   Instead, query the Manager: "What keys in group 'step' are `MISSING` for this `gen_id`?"
    *   Launch tasks for those keys.

**Verification:** Stages no longer contain logic to sync/diff keys. They simply execute the "Missing" list provided by the Manager.

### Step 5: Isolate the View Stage
**Objective:** Decouple the UI visualization (View Stage) from the Data Generation (Global ID) to allow independent zooming/panning.

**Implementation:**
1.  Introduce a `_view_generation_id` in the Pipeline, separate from the `_global_data_id`.
2.  Increment `_view_generation_id` when `set_view_context` (zoom/theme) is called.
3.  Remove `ViewStage` from the main `reconcile_all` loop.
4.  Make `ViewStage` reactive:
    *   When the Data Pipeline emits `artifact_ready` (Data ID), trigger a view render (Data ID + View ID).
    *   When the View Context changes, trigger a re-render of current data (Data ID + New View ID).

**Verification:** Zooming the canvas causes view re-renders (and increments View ID) but does *not* trigger the heavy Workpiece/Step generation logic (Global ID remains unchanged).

### Step 6: Explicit Dependency Registration
**Objective:** Replace implicit dependency checks inside stages with an explicit graph defined by the Pipeline.

**Implementation:**
1.  Add `manager.register_dependency(child_key, parent_key, gen_id)` API.
2.  Update `Pipeline.reconcile_all`:
    *   After declaring keys, explicitly register relationships (e.g., `WorkpieceKey -> StepKey`, `StepKey -> JobKey`).
3.  Update `manager.query_work_for_stage`:
    *   Only return a key as "buildable" if all its registered children for that `gen_id` are `READY`.
4.  Remove manual `if dependency.is_ready()` checks from Stages.

**Verification:** The Pipeline definition serves as the single source of truth for the execution graph. Stages automatically wait for dependencies without custom logic.

### Step 7: Generation-Based Pruning
**Objective:** Eliminate memory leaks by implementing proper garbage collection for old data.

**Implementation:**
1.  Add `manager.retain_generation(gen_id)` and `manager.release_generation(gen_id)`.
2.  Update `Pipeline`:
    *   Call `retain` on the new `_global_generation_id` at start of reconcile.
    *   Call `release` on the previous ID.
3.  Update `Manager`:
    *   When a generation's ref-count hits 0, delete all Ledger Entries and unlink all Shared Memory blocks tagged with that ID.
4.  (Optional) Allow Export jobs to `retain` a specific ID to keep it alive during long operations.

**Verification:** Monitor shared memory usage. Confirm that after modifying the document, the memory used by the previous state is freed immediately (or after the view updates catch up).