# Multi-Layer Workflow

RayForge's multi-layer system allows you to organize complex jobs into separate processing stages, each with its own operations and settings. This is essential for combining different processes like engraving and cutting, or working with multiple materials.

## What Are Layers?

A **layer** in RayForge is:

- **A container** for workpieces (imported shapes, images, text)
- **A workflow** defining how those workpieces are processed
- **An execution unit** processed sequentially during jobs

**Key concept:** Layers are processed in order, one after another, allowing you to control the sequence of operations.

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

**4. Color Layer Mapping**

Organize by color:

- **Layer 1:** All red paths (cutting layer)
- **Layer 2:** All black paths (engraving layer)
- **Layer 3:** All blue paths (scoring layer)

**Workflow:**
- Import SVG with color-coded paths
- RayForge can auto-assign paths to layers by color
- Each layer gets appropriate operation settings

---

## Creating and Managing Layers

### Adding a New Layer

1. **Click the "+" button** in the Layers panel
2. **Name the layer** descriptively (e.g., "Engrave Layer", "Cut Layer")
3. **The layer appears** in the layer list

**Default:** New documents start with one layer.

### Layer Properties

Each layer has:

| Property | Description |
|----------|-------------|
| **Name** | User-facing label |
| **Visible** | Toggle visibility in canvas and preview |
| **Stock Item** | Optional material association |
| **Workflow** | The operation(s) applied to workpieces in this layer |
| **Workpieces** | The shapes/images contained in this layer |

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
    Deleting a layer removes all its workpieces and workflow settings. Use Undo if you delete accidentally.

---

## Assigning Workpieces to Layers

### Manual Assignment

1. **Import or create** a workpiece
2. **Drag the workpiece** to the desired layer in the Layers panel
3. **Or use the properties panel** to change the workpiece's layer

### Auto-Assignment by Color

When importing SVG files:

1. **Enable "Assign by Color"** in import settings
2. **RayForge creates layers** based on stroke/fill colors
3. **Each color** gets its own layer automatically

**Example:** SVG with red, black, and blue paths:
- Red paths  "Red Layer"
- Black paths  "Black Layer"
- Blue paths  "Blue Layer"

### Moving Workpieces Between Layers

**Drag and drop:**
- Select workpiece(s) in the canvas or Document panel
- Drag to target layer in Layers panel

**Cut and paste:**
- Cut workpiece from current layer (Ctrl+X)
- Select target layer
- Paste (Ctrl+V)

---

## Layer Workflows

Each layer has a **Workflow** that defines how its workpieces are processed.

### Workflow Structure

A workflow consists of:

1. **Producer** - Generates initial toolpaths from workpieces
   - Contour, Raster, Depth Engraving, etc.

2. **Transformers** (optional) - Modify toolpaths
   - Overscan, Kerf adjustment, Tabs, etc.

3. **Post-processing** - Final optimizations
   - Path ordering, speed optimization

### Common Layer Workflows

**Engraving Layer:**
- Producer: Raster Engraving
- Settings: 300-500 DPI, moderate speed
- No transformers typically needed

**Cutting Layer:**
- Producer: Contour Cutting
- Transformers: Tabs (for holding parts), Overscan (for clean edges)
- Settings: Slower speed, higher power

**Scoring Layer:**
- Producer: Contour (without cutting through)
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

| State | Canvas | Preview | G-code |
|-------|--------|---------|--------|
| **Visible & Enabled** |  |  |  |
| **Hidden & Enabled** |  |  |  |
| **Visible & Disabled** |  |  |  |
| **Hidden & Disabled** |  |  |  |

!!! note "Disabling Layers"
    To temporarily exclude a layer from jobs without deleting it, disable the layer's workflow or remove its operation.

---

## Layer Execution Order

### How Layers are Processed

During job execution:

```
FOR each layer (in order):
  [Layer Start Hook]  (if configured)

  FOR each workpiece in layer:
    [Workpiece Start Hook]  (if configured)
    ... execute workpiece operations ...
    [Workpiece End Hook]  (if configured)

  [Layer End Hook]  (if configured)
```

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

### Conditional Layer Processing

**Using hooks for layer control:**

Layer Start Hook:
```gcode
; Layer: {layer_name}
M0 (Pause for material change)
```

**Use case:** Pause between layers to inspect, adjust focus, or change materials.

### Layer-Specific Settings

Each layer's workflow can have unique settings:

| Layer | Operation | Speed | Power | Passes |
|-------|-----------|-------|-------|--------|
| Engrave | Raster | 300 mm/min | 20% | 1 |
| Score | Contour | 500 mm/min | 10% | 1 |
| Cut | Contour | 100 mm/min | 90% | 2 |

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

---

## Related Pages

- [Operations](operations/index.md) - Operation types for layer workflows
- [Simulation Mode](simulation-mode.md) - Preview multi-layer execution
- [Macros & Hooks](macros-hooks.md) - Layer-level hooks for automation
- [3D Preview](../ui/3d-preview.md) - Visualize layer stack
