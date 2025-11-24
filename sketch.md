### Architecture Overview: Integrating the Parametric Sketcher

This document outlines the architecture for integrating a parametric sketcher into the Rayforge application. The primary goal is to treat sketches as a first-class source for creating and editing `WorkPiece` geometry, enabling a lossless, parametric workflow.

#### Guiding Principles

1.  **1:1 Relationship:** Each `Sketch` corresponds to exactly one `SourceAsset` and generates exactly one `WorkPiece`. The `WorkPiece` represents the entire solved output of its source sketch.
2.  **Sketches are a Data Source:** A sketch is a form of source data, just like an SVG or PNG. We will leverage our existing `Importer` and `SourceAsset` infrastructure to manage it.
3.  **Atomic Updates:** Editing a sketch is an atomic operation. The old geometry is completely replaced by the new solved geometry, which avoids complex and fragile geometric correspondence problems.
4.  **Non-Modal Workflow:** Sketching is a core creative task. The UI will use a dedicated, non-modal workbench widget that replaces the main canvas view, providing a focused but seamless user experience.

#### Core Components & Data Flow

The data flows from a parametric definition to a renderable object on the canvas:

`Sketch` (in-memory model) -> `SketchImporter` -> `SourceAsset` + `WorkPiece` -> `WorkPieceElement` (on-canvas view)

1.  **`Sketch` Model (`core/sketcher/`):**
    *   The in-memory representation of the parametric sketch, containing entities, parameters, and constraints.
    *   **Responsibility:** It must be fully serializable to and from a JSON-compatible dictionary for persistence.

2.  **`SketchImporter` (`image/sketch/importer.py`):**
    *   A new class inheriting from the base `Importer`. It is the canonical way to process sketch data.
    *   **Responsibility:** To take serialized sketch data (as `bytes`), deserialize it, solve it, and produce a self-contained `ImportPayload`.
    *   It will be registered to handle `.rfs` files for file-based workflows and will also be used programmatically for in-memory sketch creation.

3.  **`SourceAsset` (`core/source_asset.py`):**
    *   We use the **existing generic `SourceAsset` class**. No subclassing is needed.
    *   For a sketch, a `SourceAsset` instance will contain:
        *   `original_data`: The serialized `Sketch` definition as JSON bytes. This is the persistent source of truth.
        *   `base_render_data`: Always `None`. The solved geometry is an ephemeral result, not a static source.
        *   `renderer_name`: The string `"SketchImporter"`, which acts as a discriminator to identify this asset's origin.

4.  **`WorkPiece` (`core/workpiece.py`):**
    *   The `WorkPiece` created by the `SketchImporter` will contain a `source_segment` whose `segment_mask_geometry` is populated with the **complete, solved `Geometry`** from the sketch.
    *   **Smart Rendering Logic:** The `WorkPiece.get_vips_image()` method will be enhanced. If it finds that its `SourceAsset` has no renderable image data, it will fall back to using the existing `OpsRenderer`, passing its own vector `boundaries` to be rasterized on the fly. This makes the `WorkPiece` self-sufficient for rendering.

5.  **`SketcherWorkbench` (UI Widget):**
    *   A dedicated, non-modal `Gtk` widget that provides the complete UI for sketch editing.
    *   **Responsibility:** It contains the interactive `SketchElement`, tool palettes, constraint panels, and parameter editors. It manages the lifecycle of a sketching session (creating, editing, finishing, canceling).

#### Key Workflows

*   **Creating a New Sketch:**
    1.  The `SketcherWorkbench` is shown. The user draws and constrains a `Sketch`.
    2.  Upon finishing, the `Sketch` is serialized to bytes.
    3.  The `SketchImporter` is instantiated programmatically with these bytes.
    4.  The importer's `ImportPayload` (a new `SourceAsset` and `WorkPiece`) is added to the document.

*   **Editing an Existing Sketch:**
    1.  The application identifies that the selected `WorkPiece`'s source is a sketch (via `renderer_name == "SketchImporter"`).
    2.  The `original_data` from the `SourceAsset` is deserialized back into a `Sketch` object.
    3.  This `Sketch` is loaded into the `SketcherWorkbench` for editing.
    4.  Upon finishing, the modified `Sketch` is solved and serialized.
    5.  The system performs an **atomic replace**:
        *   The `WorkPiece`'s `segment_mask_geometry` is replaced with the new solved geometry.
        *   The `SourceAsset`'s `original_data` is replaced with the new serialized sketch definition.
    *   This preserves all user-applied transformations on the `WorkPiece` while updating its core shape.

*   **Converting Imported Vectors:**
    *   When editing a `WorkPiece` from a non-sketch source (e.g., SVG), its vector `boundaries` are converted into a new, unconstrained `Sketch`.
    *   When the user finishes editing this new sketch, a *new* `SourceAsset` is created for the sketch definition, and the *existing* `WorkPiece` is re-linked to point to this new asset.



### Iterative Implementation Plan

#### Step 1: Create the Base Exporter and Sketch Exporter

*   **Goal:** Establish a formal pattern for exporting data and implement it for the `Sketch` model. This creates the `.rfs` file format.
*   **Tasks:**
    1.  **Define `BaseExporter` (`rayforge/image/base_exporter.py`):**
        *   Create an abstract base class `Exporter`.
        *   The constructor `__init__(self, doc_item: DocItem)` will take the item to be exported.
        *   An abstract method `export() -> bytes` will be the main entry point.
        *   Define metadata properties: `label`, `extensions`, `mime_types`.
    2.  **Implement `SketchExporter` (`rayforge/image/sketch/exporter.py`):**
        *   Create `class SketchExporter(Exporter)`.
        *   The constructor will expect a `WorkPiece` whose source is a sketch (for now, we'll fake this in tests).
        *   The `export()` method will:
            a. Check if the `WorkPiece`'s source is a sketch.
            b. Retrieve the serialized `Sketch` bytes from `source.original_data`.
            c. Return these bytes.
    3.  **Create Test Suite:**
        *   Write a test for the `SketchExporter` (`tests/image/sketch/test_sketch_exporter.py`):
            a. Manually create a `Sketch` object in memory and serialize it to bytes.
            b. Create a mock `SourceAsset` and `WorkPiece` that use these bytes.
            c. Pass this mock `WorkPiece` to the `SketchExporter`.
            d. Assert that `exporter.export()` returns the exact same bytes.
*   **Testability:** The backend is now capable of producing a valid `.rfs` file format from an in-memory `Sketch`. The main application is completely unaffected.

#### Step 2: Create the Sketch Importer and Test Round-Trip

*   **Goal:** Implement the logic to parse a `.rfs` file and test the entire serialization-deserialization loop.
*   **Tasks:**
    1.  **Implement `SketchImporter` (`rayforge/image/sketch/importer.py`):**
        *   Create `class SketchImporter(Importer)`.
        *   Set its metadata (`label`, `extensions`, etc.).
        *   The `get_doc_items()` method will **initially be simplified**. For this step, it should parse the `raw_data` into a `Sketch` object and store it in a temporary, non-document property like `self.parsed_sketch`. It should return `None` or an empty payload, as we are not creating workpieces yet. This keeps the test focused.
    2.  **Create Test Suite (`tests/image/sketch/test_sketch_importer.py`):**
        *   Create a test that performs a full round-trip:
            a. Create a complex `Sketch` object.
            b. Use a (hypothetical) `SketchExporter` to get its `bytes`.
            c. Pass these `bytes` to a `SketchImporter`.
            d. Call the importer's simplified `get_doc_items()`.
            e. Assert that `importer.parsed_sketch` is a valid `Sketch` object identical to the original.
*   **Testability:** You can now programmatically convert a `Sketch` to bytes and back again with confidence. This confirms the file format is stable. The main app is still untouched.

#### Step 3: Integrate a Standalone Sketch Workbench UI

*   **Goal:** Get the sketcher UI running in a "sandbox" mode, completely separate from the main document, allowing users to create, open, and save `.rfs` files.
*   **Tasks:**
    1.  **Create the `SketcherWorkbench` Widget:** This is the container for the `SketchElement`, tool palettes, constraint buttons, and parameter lists.
    2.  **Add Menu Items:** Add "New Sketch", "Open Sketch...", and "Save Sketch As..." to the main application menu.
    3.  **Implement Window Logic:** These menu items will open a **new, separate `Gtk.Window`** that contains only the `SketcherWorkbench`. This window manages its own `Sketch` object.
    4.  **Wire Up I/O:**
        *   "New Sketch" creates a new `SketcherWorkbench` with a fresh `Sketch` model.
        *   "Open Sketch..." shows a file dialog. On success, it reads the file bytes, uses the `SketchImporter` to get a `Sketch` object, and loads it into a new `SketcherWorkbench`.
        *   "Save Sketch As..." gets the current `Sketch` from the active `SketcherWorkbench`, uses a conceptual `SketchExporter` to get its bytes, and writes them to a file.
*   **Testability:** The application now has a fully functional, sandboxed sketch editor. Users can create a sketch, save it, close the window, open it again, and see their work perfectly restored. This has **zero impact** on the main document canvas and its workflows.

#### Step 4: Implement the Sketch Renderer Logic

*   **Goal:** Enable a `WorkPiece` that has vector data but no image data to render itself correctly. This is a critical prerequisite for the final integration.
*   **Tasks:**
    1.  **Modify `WorkPiece.get_vips_image()`** (or its equivalent rendering entry point, like `WorkPieceElement.render_to_surface`).
    2.  Implement the "smart rendering" logic:
        a. Check if the `SourceAsset` provides renderable image data (`base_render_data` or decodable `original_data`).
        b. **If not**, check if `self.source_segment.segment_mask_geometry` contains valid vectors.
        c. **If it does**, invoke the existing `OpsRenderer`, passing `self.boundaries` to it.
*   **Testability:** Write a crucial unit test. Manually construct a `WorkPiece` with a `SourceAsset` that has no image data but a `SourceAssetSegment` that *does* have geometry. Call `get_vips_image()` on it and assert that it returns a valid, non-empty `pyvips.Image` by using the `OpsRenderer` fallback path.

#### Step 5: Allow Adding Sketches to the Document ("Finish Sketch")

*   **Goal:** Bridge the gap between the sandbox sketcher and the main document.
*   **Tasks:**
    1.  **Update `SketchImporter.get_doc_items()`:** Modify it to the full implementation. It must now solve the sketch and return a complete `ImportPayload` containing a `SourceAsset` and a `WorkPiece`. The `SourceAsset`'s `renderer_name` should be set to `"SketchImporter"`.
    2.  **Add View Switching:** Replace the "new window" logic from Step 3. The "New Sketch" action should now hide the main `WorkSurface` and show the `SketcherWorkbench` in its place.
    3.  **Implement "Finish Sketch" Button:**
        a. This button will serialize the current `Sketch` to bytes.
        b. It will instantiate `SketchImporter(data=sketch_bytes, source_file=None)`.
        c. It will call `importer.get_doc_items()` to get the payload.
        d. It will add the `SourceAsset` and `WorkPiece` from the payload to the active `Doc`.
        e. It will switch the view back to the main `WorkSurface`.
*   **Testability:** This is the first full integration test. Create a new sketch. Click "Finish Sketch". The view switches back, and a new, correctly rendered `WorkPieceElement` appears on the main canvas. You can save and load the document, and the sketch-based workpiece persists.

#### Step 6: Allow Editing Document Sketches

*   **Goal:** Enable a round-trip workflow for sketches that are already part of the document.
*   **Tasks:**
    1.  **Add "Edit Sketch" UI:** Add a context menu item or button that is only enabled when a selected `WorkPiece`'s `source.renderer_name == "SketchImporter"`.
    2.  **Implement Edit Logic:**
        a. On click, get the `original_data` (sketch bytes) from the `WorkPiece`'s `SourceAsset`.
        b. Deserialize it into a `Sketch` object.
        c. Switch to the `SketcherWorkbench` view, loading this `Sketch` into its `SketchElement`.
    3.  **Modify "Finish Sketch" Logic:** Add a branch. If the workbench was opened for editing (pass a flag or the `WorkPiece` being edited), it must perform an **update** instead of a create:
        a. Find the original `WorkPiece` and `SourceAsset`.
        b. Replace `SourceAsset.original_data` with the new serialized sketch.
        c. Re-solve the sketch and replace the `WorkPiece.source_segment.segment_mask_geometry` with the new result.
*   **Testability:** Create a sketch workpiece. Select it, edit it, and finish. The workpiece on the main canvas should update to the new shape. Its position/rotation should be preserved.

#### Step 7: Allow Converting Workpieces to Sketches

*   **Goal:** Enable editing of imported vector shapes using the sketcher.
*   **Tasks:**
    1.  **Enable "Edit Sketch" for Vectors:** The UI action should also be enabled for any non-sketch `WorkPiece` that has vector `boundaries`.
    2.  **Implement Conversion Logic:** In the "Edit Sketch" handler, if the `renderer_name` is *not* "SketchImporter":
        a. Show a "one-way conversion" warning.
        b. Get `workpiece.boundaries`.
        c. Use a new utility function `geometry_to_sketch(geometry)` to create a new, unconstrained `Sketch`.
        d. Load this new `Sketch` into the `SketcherWorkbench`.
    3.  **Modify "Finish Sketch" Logic for Conversions:** When finishing an edit that started as a conversion:
        a. Create a *new* `SourceAsset` for the sketch data.
        b. Update the *existing* `WorkPiece`'s `source_segment` to point to this new `SourceAsset`'s UID and contain the new solved geometry.
*   **Testability:** Import an SVG. Select it and "Edit Sketch". Add constraints. Finish. The workpiece updates. Edit it *again*. This time, it should open the constrained sketch you just made, proving the source was successfully re-linked.