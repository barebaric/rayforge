### Final Architecture: Depth Engraving with a Decoupled Path Utility Module

This architecture introduces a high-performance `DepthEngraver` producer. It is built upon a cleaner codebase where complex, stateless path algorithms are refactored out of the `Ops` and `Geometry` classes into a shared, generic `core/path/` utility module.

#### 1. Core Feature: The `DepthEngraver` OpsProducer

*   **Location:** `.../pipeline/opsproducer/depth.py`
*   **Purpose:** To translate grayscale image data into laser engraving paths.
*   **Strategies:** It will be built with a strategy pattern to support different engraving methods.
    *   **Initial Strategy: `VARY_POWER`:** This will be the first implementation. It scans the `WorkPiece`'s rendered surface in its **local coordinate system** and generates a series of axis-aligned `ScanLinePowerCommand`s, one for each row of pixels.

#### 2. Performance Keystone: The `ScanLinePowerCommand`

*   **Location:** `.../core/ops/commands.py`
*   **Purpose:** A new, highly efficient `MovingCommand` that represents a full raster scan line with continuously varying power. This avoids the overhead of creating thousands of individual `SetPower`/`LineTo` commands.
*   **Structure:** It will contain `start_point`, `end`, and a `power_values: bytearray`.
*   **Transformer Compatibility:** The command class itself will contain the necessary logic to be compatible with the existing `OpsTransformer` pipeline.

#### 3. New `MovingCommand` Interfaces

To ensure all commands, including the new `ScanLinePowerCommand`, can be correctly handled by transformers, two methods will be added to the `MovingCommand` base class and implemented by its children:

*   **`flip()`:** Allows the `Optimize` transformer to reverse the direction of any path segment by calling `cmd.flip()`. The command itself knows how to correctly reconfigure its internal data (e.g., reverse its `power_values`, recalculate arc offsets).
*   **`linearize()`:** Allows transformers like `TabOpsTransformer` to deconstruct complex commands into a sequence of simple lines for geometric analysis, without needing to know the implementation details of every command type.

#### 4. The `core/path/` Utility Module

*   **Purpose:** To house shared, stateless, low-level path manipulation algorithms, eliminating code duplication between `Ops` and `Geometry`.
*   **Structure:** A collection of pure Python functions in modules like `core/path/linearize.py`, `core/path/geometry.py`, etc.
*   **Content:** It will contain functions like `linearize_arc()` and `get_bounding_rect()`. These functions are generic and have no knowledge of `Ops` or `Geometry` objects.
*   **Usage:** The `Ops` and `Geometry` classes will call these utility functions to implement their high-level methods, thus simplifying their own code while retaining their unique "business logic."

---

### Iterative Implementation Plan

#### Phase 1: Foundational Refactoring (The `path` Module)

1.  **Create `path` Module:** Create the `core/path/` directory and initial files (e.g., `linearize.py`).
2.  **Extract `linearize_arc`:**
    *   Move the `_linearize_arc` logic from `Ops` into a new public function `linearize_arc` in `core/path/linearize.py`.
    *   Refactor the `Ops` and `Geometry` classes to delete their private `_linearize_arc` methods and call the new shared utility function instead.
3.  **Test:** Run the entire existing test suite. All tests involving arc linearization for both `Ops` and `Geometry` must pass.
4.  **Iteratively Extract:** Repeat the process for other stateless utility candidates (e.g., bounding box math), moving the core logic, updating `Ops`/`Geometry` to call the new helper, and testing after each move.

#### Phase 2: Enhance the `ops` Module for the New Feature

1.  **Refactor `ops.py`:** Split the `ops.py` file into `core/ops/container.py` (for the `Ops` class) and `core/ops/commands.py` (for `Command` subclasses and `State`). Create `__init__.py` to maintain the public API.
2.  **Implement `ScanLinePowerCommand`:** Add the new `ScanLinePowerCommand` class to `core/ops/commands.py`.
3.  **Implement `flip()` and `linearize()` Interfaces:**
    *   Add the abstract methods to the `MovingCommand` base class in `commands.py`.
    *   Provide concrete implementations for `LineToCommand`, `ArcToCommand`, and the new `ScanLinePowerCommand`.
4.  **Unit Test:** Write specific unit tests for the new command and its interface methods, verifying correct behavior for flipping and linearization.

#### Phase 3: Update Transformers for Compatibility

1.  **Update `Optimize`:** Refactor `optimize.py:flip_segment` to be a simpler function that iterates through a reversed segment and calls `cmd.flip()` on each command. This removes its internal `isinstance` logic and makes it future-proof.
2.  **Update `TabOpsTransformer`:** Refactor `ops.container.Ops.subtract_regions` to call `cmd.linearize()` on every command it processes. This will make it capable of applying tabs to any linearizable command, including the new raster lines.
3.  **Update Vector Transformers (`Smooth`, `ArcWeld`):** Modify their segment-identification logic to explicitly reject segments containing `ScanLinePowerCommand`, ensuring they only process vector geometry.
4.  **Test:** Run all transformer-related tests to confirm that they work correctly with the new, more generic logic and safely ignore raster data where appropriate.

#### Phase 4: Implement the `DepthEngraver` Producer

1.  **Create Producer File:** Create `.../pipeline/opsproducer/depth.py` with a new `DepthEngraver(OpsProducer)` class.
2.  **Implement `VARY_POWER` Strategy:** In the `run()` method, implement the logic to:
    *   Get the rendered `cairo.surface` from the `WorkPiece`.
    *   Convert the surface to a grayscale NumPy array.
    *   Iterate through the array row by row.
    *   For each row, create a `power_values: bytearray`.
    *   Create a single, horizontal `ScanLinePowerCommand` using the row's local coordinates and the power bytearray.
    *   Return an `Ops` object containing the list of these commands.
3.  **Register Producer:** Add `DepthEngraver` to the `producer_by_name` dictionary in `.../pipeline/opsproducer/__init__.py`.

#### Phase 5: Implement G-Code Generation

1.  **Update G-Code Generator:** Add a new handler that recognizes `ScanLinePowerCommand`.
2.  **Implement Logic:** When it encounters the command, it must:
    *   Note that the command's `start` and `end` may now be for a diagonal line due to transformations.
    *   Calculate the total length and the small vector for a single "pixel" step (`(end - start) / len(power_values)`).
    *   Loop through the `power_values` array, emitting an interleaved `S<power>` and `G1 X... Y...` command for each step.

#### Phase 6: UI Integration

1.  **Add New Step Type:** Update the application UI (e.g., a "New Step" menu) to include a "Depth Engrave" option that creates a `Step` configured with the `DepthEngraver` producer.
2.  **Update `StepSettingsDialog`:**
    *   Add logic to show a new "Depth Engraving Settings" `Adw.PreferencesGroup` when the current step is a `DepthEngraver` type.
    *   Add UI controls (e.g., `Adw.SpinRow` or sliders) for parameters like "Min Power %" and "Max Power %".
    *   Bind these controls to the `step.opsproducer_dict["params"]` dictionary, using `DictItemCommand` for undo/redo support.