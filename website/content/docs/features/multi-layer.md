# Multi-Layer Workflow

Rayforge's multi-layer system allows you to organize complex jobs into separate processing stages, each with its own operations and settings. This is essential for combining different processes like engraving and cutting, or working with multiple materials.

## What Are Layers?

A **layer** in Rayforge is:

- **A container** for workpieces (imported shapes, images, text)
- **A workflow** defining how those workpieces are processed
- **A step** processed sequentially during jobs

**Key concept:** Layers are processed in order, one after another, allowing
you to control the sequence of operations.

!!! note "Layers and Workpieces"
    A layer contains one or more workpieces. When importing SVG files with
    layers, each layer from your design becomes a separate layer in Rayforge.
    This lets you keep your design organized exactly as you created it.

---

## Why Use Multiple Layers?

### Common Use Cases

**1. Engrave then Cut**

The most common multi-layer workflow:

- **Layer 1:** Raster engrave the design
- **Layer 2:** Contour cut the outline

**Why separate layers?**

- Engraving first ensures the piece doesn't move during engraving
- Cutting last prevents pieces from falling before engraving completes
- Different power/speed settings for each operation

**2. Multi-Pass Cutting**

For thick materials:

- **Layer 1:** First pass at moderate power
- **Layer 2:** Second pass at full power (same geometry)
- **Layer 3:** Optional third pass if needed

**Benefits:**

- Reduces charring compared to single high-power pass
- Each layer can have different speed/power settings

**3. Multi-Material Projects**

Different materials in one job:

- **Layer 1:** Cut acrylic parts
- **Layer 2:** Engrave wood parts
- **Layer 3:** Mark metal parts

**Requirements:**

- Each layer targets different areas of the bed
- Different speed/power/focus for each material

**4. SVG Layer Import**

Import SVG files with existing layer structure:

- **Layer 1:** Engraving elements from SVG
- **Layer 2:** Cutting elements from SVG
- **Layer 3:** Scoring elements from SVG

**Workflow:**

- Import an SVG file that has layers
- Enable "Use Original Vectors" in the import dialog
- Select which layers to import from the detected layers list
- Each layer becomes a separate layer in Rayforge

**Requirements:**

- Your SVG file must use layers (created in Inkscape or similar software)
- Enable "Use Original Vectors" when importing
- Layer names are preserved from your design software

---

## Creating and Managing Layers

### Adding a New Layer

1. **Click the "+" button** in the Layers panel
2. **Name the layer** descriptively (e.g., "Engrave Layer", "Cut Layer")
3. **The layer appears** in the layer list

**Default:** New documents start with one layer.

### Layer Properties

Each layer has:

| Property       | Description                                          |
| -------------- | ---------------------------------------------------- |
| **Name**       | The name shown in the layer list                     |
| **Visible**    | Toggle visibility in canvas and preview              |
| **Stock Item** | Optional material association                        |
| **Workflow**   | The operation(s) applied to workpieces in this layer |
| **Workpieces** | The shapes/images contained in this layer            |

!!! note "Layers as Containers"
    Layers are containers for your workpieces. When importing SVG files with
    layers, each layer from your design becomes a separate layer in Rayforge.

### Reordering Layers

**Execution order = layer order in the list (top to bottom)**

To reorder:

1. **Drag and drop** layers in the Layers panel
2. **Order matters** - layers execute from top to bottom

**Example:**

```
Layers Panel:
1. Engrave Layer     Executes first
2. Score Layer       Executes second
3. Cut Layer         Executes last (recommended)
```

### Deleting Layers

1. **Select the layer** in the Layers panel
2. **Click the delete button** or press Delete
3. **Confirm deletion** (all workpieces in the layer are removed)

!!! warning "Deletion is Permanent"
    Deleting a layer removes all its workpieces and workflow settings. Use Undo
    if you delete accidentally.

---

## Assigning Workpieces to Layers

### Manual Assignment

1. **Import or create** a workpiece
2. **Drag the workpiece** to the desired layer in the Layers panel
3. **Or use the properties panel** to change the workpiece's layer

### SVG Layer Import

When importing SVG files with "Use Original Vectors" enabled:

1. **Enable "Use Original Vectors"** in the import dialog
2. **Rayforge detects layers** from your SVG file
3. **Select which layers** to import using the layer switches
4. **Each selected layer** becomes a separate layer with its own workpiece

!!! note "Layer Detection"
    Rayforge automatically detects layers from your SVG file. Each layer
    you created in your design software will appear as a separate layer in
    Rayforge.

!!! note "Vector Import Only"
    Layer selection is only available when using direct vector import.
    When using trace mode, the entire SVG is processed as one workpiece.

### Moving Workpieces Between Layers

**Drag and drop:**

- Select workpiece(s) in the canvas or Document panel
- Drag to target layer in Layers panel

**Cut and paste:**

- Cut workpiece from current layer (Ctrl+X)
- Select target layer
- Paste (Ctrl+V)

### SVG Import Dialog

When importing SVG files, the import dialog provides options that affect
layer handling:

**Import Mode:**

- **Use Original Vectors:** Preserves your vector paths and layer structure.
  When enabled, a "Layers" section appears showing all layers from your file.
- **Trace Mode:** Converts the SVG to a bitmap and traces the outlines.
  Layer selection is disabled in this mode.

**Layers Section (Vector Import Only):**

- Shows all layers from your SVG file
- Each layer has a toggle switch to enable/disable import
- Layer names from your design software are preserved
- Only selected layers are imported as separate layers

!!! tip "Preparing SVG Files for Layer Import"
    To use SVG layer import, create your design with layers in software like
    Inkscape. Use the Layers panel to organize your design, and Rayforge
    will preserve that structure.

---

## Layer Workflows

Each layer has a **Workflow** that defines how its workpieces are processed.

### Setting Up Layer Workflows

For each layer, you choose an operation type and configure its settings:

**Operation Types:**

- **Contour** - Follows outlines (for cutting or scoring)
- **Raster Engraving** - Engraves images and fills areas
- **Depth Engraving** - Creates varying depth engravings

**Optional Enhancements:**

- **Tabs** - Small bridges to hold parts in place during cutting
- **Overscan** - Extends cuts beyond the shape for cleaner edges
- **Kerf Adjustment** - Compensates for the laser's cutting width

### Common Layer Setups

**Engraving Layer:**

- Operation: Raster Engraving
- Settings: 300-500 DPI, moderate speed
- Typically no additional options needed

**Cutting Layer:**

- Operation: Contour Cutting
- Options: Tabs (to hold parts), Overscan (for clean edges)
- Settings: Slower speed, higher power

**Scoring Layer:**

- Operation: Contour (light power, doesn't cut through)
- Settings: Low power, fast speed
- Purpose: Fold lines, decorative lines

---

## Layer Visibility

Control which layers are shown in the canvas and previews:

### Canvas Visibility

- **Eye icon** in Layers panel toggles visibility
- **Hidden layers:**
  - Not shown in 2D canvas
  - Not shown in 3D preview
  - **Still included in generated G-code**

**Use cases:**

- Hide complex engraving layers while positioning cut layers
- Declutter the canvas when working on specific layers
- Focus on one layer at a time

### Visibility vs. Enabled

| State                  | Canvas | Preview | G-code |
| ---------------------- | ------ | ------- | ------ |
| **Visible & Enabled**  |        |         |        |
| **Hidden & Enabled**   |        |         |        |
| **Visible & Disabled** |        |         |        |
| **Hidden & Disabled**  |        |         |        |

!!! note "Disabling Layers"
To temporarily exclude a layer from jobs without deleting it, turn off the
layer's operation or disable it in the layer settings.

---

## Layer Execution Order

### How Layers are Processed

During job execution, Rayforge processes each layer in order from top to
bottom. Within each layer, all workpieces are processed before moving to
the next layer.

### Order Matters

**Wrong order:**

```
1. Cut Layer
2. Engrave Layer
```

**Problem:** Cut parts may fall out or move before engraving!

**Correct order:**

```
1. Engrave Layer
2. Cut Layer
```

**Why:** Engraving happens while part is still attached, then cutting frees it.

### Multiple Passes

For thick materials, create multiple cutting layers:

```
1. Engrave Layer
2. Cut Layer (Pass 1) - 50% power
3. Cut Layer (Pass 2) - 75% power
4. Cut Layer (Pass 3) - 100% power
```

**Tip:** Use the same geometry for all cutting passes (duplicate the layer).

---

## Advanced Techniques

### Layer Grouping by Material

Use layers to organize by material when running mixed jobs:

```
Material 1 (3mm Acrylic):
  - Acrylic Engrave Layer
  - Acrylic Cut Layer

Material 2 (3mm Plywood):
  - Wood Engrave Layer
  - Wood Cut Layer
```

**Workflow:**

1. Process all Material 1 layers
2. Swap materials
3. Process all Material 2 layers

**Alternative:** Use separate documents for different materials.

### Pausing Between Layers

You can configure Rayforge to pause between layers. This is useful when you
need to:

- Change materials mid-job
- Inspect progress before continuing
- Adjust focus for different operations

To set up layer pauses, use the hooks feature in your machine settings.

### Layer-Specific Settings

Each layer's workflow can have unique settings:

| Layer   | Operation | Speed      | Power | Passes |
| ------- | --------- | ---------- | ----- | ------ |
| Engrave | Raster    | 300 mm/min | 20%   | 1      |
| Score   | Contour   | 500 mm/min | 10%   | 1      |
| Cut     | Contour   | 100 mm/min | 90%   | 2      |

---

## Best Practices

### Naming Conventions

**Good layer names:**

- "Engrave - Logo"
- "Cut - Outer Contour"
- "Score - Fold Lines"
- "Pass 1 - Rough Cut"
- "Pass 2 - Final Cut"

**Poor layer names:**

- "Layer 1", "Layer 2" (not descriptive)
- Long descriptions (keep concise)

### Layer Organization

1. **Top to bottom = execution order**
2. **Engraving before cutting** (general rule)
3. **Group related operations** (all cutting, all engraving)
4. **Use visibility** to focus on current work
5. **Delete unused layers** to keep projects clean

### Preparing SVG Files for Layer Import

**For best results when importing SVG layers:**

1. **Use the Layers panel** in your design software to organize your design
2. **Assign meaningful names** to each layer (e.g., "Engrave", "Cut")
3. **Keep layers flat** - avoid putting layers inside other layers
4. **Save your file** and import into Rayforge
5. **Verify layer detection** by checking the import dialog

Rayforge works best with SVG files created in Inkscape or similar vector
design software that supports layers.

### Performance

**Many layers:**

- No significant performance impact
- 10-20 layers is common for complex jobs
- Organize logically, not to minimize layer count

**Simplify if needed:**

- Combine similar operations into one layer when possible
- Use fewer raster engravings (most resource-intensive)

---

## Troubleshooting

### Layer Not Generating G-code

**Problem:** Layer appears in document but not in generated G-code.

**Solutions:**

1. **Check layer has workpieces** - Empty layers are skipped
2. **Check workflow is configured** - Layer needs an operation
3. **Verify operation settings** - Power > 0, valid speed, etc.
4. **Check workpiece visibility** - Hidden workpieces may not process
5. **Regenerate G-code** - Make a small change to force regeneration

### Wrong Layer Order

**Problem:** Operations execute in unexpected order.

**Solution:** Reorder layers in the Layers panel. Remember: top = first.

### Layers Overlapping in Preview

**Problem:** Multiple layers show overlapping content in preview.

**Clarification:** This is normal if layers share the same XY area.

**Solutions:**

- Use layer visibility to hide other layers temporarily
- Check 3D preview to see depth/order
- Verify this is intentional (e.g., engraving then cutting same shape)

### Workpiece in Wrong Layer

**Problem:** Workpiece was assigned to incorrect layer.

**Solution:** Drag workpiece to correct layer in Layers panel or Document tree.

### SVG Layers Not Detected

**Problem:** Importing an SVG file but no layers appear in the import dialog.

**Solutions:**

1. **Check SVG structure** - Open your file in Inkscape or similar software
   to verify it has layers
2. **Enable "Use Original Vectors"** - Layer selection is only available in
   this import mode
3. **Verify your design has layers** - Make sure you created layers in your
   design software, not just groups
4. **Check for nested layers** - Layers inside other layers may not be
   detected properly
5. **Re-save your file** - Sometimes re-saving with a current version of
   your design software helps

### SVG Layer Import Shows Wrong Content

**Problem:** Imported layer shows content from other layers or is empty.

**Solutions:**

1. **Check layer selection** - Verify the correct layers are enabled in the
   import dialog
2. **Verify your design** - Open the original file in your design software
   to confirm each layer contains the right content
3. **Check for shared elements** - Elements that appear in multiple layers
   may cause confusion
4. **Try trace mode** - Use trace mode as a fallback if vector import has
   issues

---

## Related Pages

- [Operations](operations/index.md) - Operation types for layer workflows
- [Simulation Mode](simulation-mode.md) - Preview multi-layer execution
- [Macros & Hooks](macros-hooks.md) - Layer-level hooks for automation
- [3D Preview](../ui/3d-preview.md) - Visualize layer stack
