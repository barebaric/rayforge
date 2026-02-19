# Holding Tabs

Holding tabs (also called bridges or tabs) are small uncut sections left along cut paths that keep parts attached to the surrounding material. This prevents cut pieces from moving during the job, which could cause misalignment, damage, or fire hazards.

## Why Use Holding Tabs?

When cutting through material, the cut piece can:

- **Shift position** mid-job, causing subsequent operations to misalign
- **Fall through** the bed grating or tilt if supported only at edges
- **Collide with** the laser head as it moves
- **Catch fire** if it falls onto hot scrap below
- **Get damaged** from falling or vibration

Holding tabs solve these problems by keeping the piece attached until you're ready to remove it.

---

## How Holding Tabs Work

Rayforge implements tabs by creating **small gaps in the cut path**:

1. You mark positions along the cut path where tabs should be
2. During G-code generation, Rayforge interrupts the cut at each tab
3. The laser lifts (or turns off), skips the tab width, then resumes cutting
4. After the job completes, you manually break or cut the tabs to free the piece

---

## Adding Holding Tabs

### Quick Add

1. **Select the workpiece** you want to add tabs to (must be a cut/contour operation)
2. **Click the tab tool** in the toolbar or press the tab shortcut
3. **Click on the path** where you want tabs:
   - Tabs appear as small handles on the path outline
   - Click multiple times to add more tabs
   - Typically 3-4 tabs for small parts, more for larger pieces
4. **Enable tabs** if not already enabled (toggle in the properties panel)

### Using the Add Tabs Popover

For more control:

1. **Right-click** on the workpiece or use **Edit → Add Tabs**
2. **Choose tab placement method:**
   - **Manual:** Click individual locations
   - **Equidistant:** Automatically space tabs evenly around the path
3. **Configure tab settings:**
   - **Number of tabs:** How many tabs to create (for equidistant)
   - **Tab width:** Length of each uncut section (typically 2-5mm)
4. **Click Apply**

---

## Tab Properties

### Tab Width

The **width** is the length of the uncut section along the path.

**Recommended widths:**

| Material | Thickness | Tab Width |
|----------|-----------|-----------|
| **Cardboard** | 1-3mm | 2-3mm |
| **Plywood** | 3mm | 3-4mm |
| **Plywood** | 6mm | 4-6mm |
| **Acrylic** | 3mm | 2-3mm |
| **Acrylic** | 6mm | 3-5mm |
| **MDF** | 3mm | 3-4mm |
| **MDF** | 6mm | 5-7mm |

**Guidelines:**
- **Thicker materials** need wider tabs for strength
- **Heavier parts** need more and/or wider tabs
- **Brittle materials** (acrylic) can use smaller tabs (easier to break)
- **Fibrous materials** (wood) may need wider tabs

:::warning Tab Width vs Material Thickness
Tabs must be wide enough to support the part but small enough to remove cleanly. Too narrow = part may break free; too wide = difficult to remove or damages the part.
:::

### Tab Position

Tabs are positioned using two parameters:

- **Segment index:** Which line/arc segment of the path
- **Position (0.0 - 1.0):** Where along that segment (0 = start, 1 = end)

**Manual placement tips:**
- Place tabs on **straight sections** when possible (easier to remove)
- Avoid tabs on **tight curves** (stress concentration)
- Distribute tabs **evenly** around the part
- Place tabs on **corners** for maximum support if needed

### Equidistant Tabs

The **equidistant** feature automatically places tabs at even intervals:

**Benefits:**
- Even weight distribution
- Predictable breaking pattern
- Quick setup for regular shapes

---

## Working with Tabs

### Editing Tabs

**Move a tab:**
1. Select the workpiece
2. Drag the tab handle along the path
3. Release to set new position

**Resize a tab:**
- Use the properties panel to adjust width
- All tabs on a workpiece share the same width

**Delete a tab:**
1. Click the tab handle to select it
2. Press Delete or use the context menu
3. Or clear all tabs and start over

### Enabling/Disabling Tabs

Toggle tabs on/off without deleting them:

- **Workpiece properties panel:** "Enable Tabs" checkbox
- **Toolbar:** Tab visibility toggle icon

**When disabled:**
- Tabs are not generated in G-code
- Tab handles are hidden in the canvas
- The path cuts completely through

**Use case:** Temporarily disable tabs to test the cut, then re-enable for production.

---

## Removing Tabs After Cutting

**Tools:**
- Craft knife or box cutter
- Flush-cut pliers
- Chisel (for wood)
- Fine saw for thick materials

**Technique:**

1. **Score the tab** from both sides if accessible
2. **Gently bend** the part to stress the tab
3. **Cut through** the remaining material
4. **Sand or file** the tab remnant flush with the edge

**For brittle materials (acrylic):**
- Use minimal tabs (they snap easily)
- Score deeply before snapping
- Support the part while breaking tabs to avoid cracks

**For wood:**
- Tabs may require cutting (don't snap cleanly)
- Use a sharp knife or chisel
- Cut flush, then sand smooth

---

## Related Pages

- [Contour Cutting](operations/contour) - Primary operation that uses tabs
- [Multi-Layer Workflow](multi-layer) - Managing tabs across multiple layers
- [3D Preview](../ui/3d-preview) - Visualizing tabs in preview
- [Simulation Mode](simulation-mode) - Previewing cuts with tab gaps
