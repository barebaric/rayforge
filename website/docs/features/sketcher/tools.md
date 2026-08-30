---
description: "Sketcher tools, keyboard shortcuts, pie menu, construction mode, grid, snapping, offset, chamfer, and fillet in Rayforge."
---

# Sketcher Tools

## Keyboard Shortcuts

The sketcher provides keyboard shortcuts for efficient workflow:

### Tool Shortcuts
- `Space`: Select tool
- `G+P`: Path tool (lines and bezier curves)
- `G+A`: Arc tool
- `G+C`: Ellipse tool
- `G+R`: Rectangle tool
- `G+O`: Rounded Rectangle tool
- `G+F`: Fill Area tool
- `G+T`: Text Box tool
- `G+Y`: Circular Array tool
- `G+W`: Array Along Curve tool
- `G+G`: Grid tool (create a grid of copies from the selection)
- `G+N`: Toggle construction mode on selection

### Action Shortcuts
- `O+F`: Offset the selected contour
- `C+H`: Add Chamfer corner
- `C+F`: Add Fillet corner
- `C+S`: Straighten selected bezier curves to lines
- `M+V`: Mirror selection vertically
- `M+H`: Mirror selection horizontally
- `Ctrl+D`: Duplicate selection in-place

### Constraint Shortcuts
- `H`: Apply Horizontal constraint
- `V`: Apply Vertical constraint
- `N`: Apply Perpendicular constraint
- `T`: Apply Tangent constraint
- `E`: Apply Equal constraint
- `O` or `C`: Apply Alignment constraint (Coincident)
- `S`: Apply Symmetry constraint
- `K+D`: Apply Distance constraint
- `K+R`: Apply Radius constraint
- `K+O`: Apply Diameter constraint
- `K+A`: Apply Angle constraint
- `K+X`: Apply Aspect Ratio constraint

### General Shortcuts
- `Ctrl+Z`: Undo
- `Ctrl+Y` or `Ctrl+Shift+Z`: Redo
- `Ctrl+D`: Duplicate selected elements
- `Delete`: Delete selected elements
- `Arrow keys`: Nudge selected entities (hold `Shift` for a larger step)
- `Escape`: Cancel current operation or deselect
- `F`: Fit view to content

## Mirror, Duplicate, and Nudge

Several transformation tools work on the current selection:

- **Mirror Vertically / Horizontally** (`M+V` / `M+H`): mirror the
  selection in-place across its bounding-box center. Constraints that
  span the selection boundary are dropped; internal constraints are
  preserved.
- **Duplicate** (`Ctrl+D`): copy the selection in-place. The copies get
  fresh IDs and remapped internal constraints; only the copies remain
  selected afterwards. Undo removes them.
- **Nudge**: with entities selected, the **arrow keys** move the
  selection. Hold `Shift` for a larger nudge step.

These are available from the toolbar and the **Sketch** menu.

## Construction Mode

Construction mode allows you to mark entities as "construction geometry" - helper
elements used to guide your design but not part of the final output. Construction
entities are displayed differently (typically as dashed lines) and are not
included when the sketch is used for laser cutting or engraving.

To toggle construction mode:
- Select one or more entities
- Press `N` or `G+N`, or use the Construction option in the pie menu

Construction entities are useful for:
- Creating reference lines and circles
- Defining temporary geometry for alignment
- Building complex shapes from a framework of guides

## Visibility Controls

The grid adapts to the zoom level and is always available as a sizing
reference; how snapping works is described in [the sketcher
overview](index.md#grid-and-snapping).

The sketcher toolbar includes toggle buttons to control visibility:

- **Show/hide construction geometry**: Toggle visibility of construction entities
- **Show/hide constraints**: Toggle visibility of constraint markers

These controls help reduce visual clutter when working on complex sketches.

### Auto-Constrain During Creation

Many drawing tools automatically apply constraints as you create geometry.
The path tool creates horizontal and vertical constraints when snap guides
show alignment during drawing, which helps keep your sketch tidy from the
start, rather than fixing things up afterward.

### Axis-Constrained Movement

When dragging points or geometry, hold `Shift` to constrain movement to the
nearest axis (horizontal or vertical). This is useful for maintaining alignment
while making adjustments.

## Offset Contour

The offset tool grows or shrinks a selected contour by a given distance, or
expands an open path into a slot. Select the entities that form a contour
(or use double-click to select connected geometry), then press `O+F`, or
use the **Offset** entry in the pie menu.

![Offset Contour dialog](/screenshots/addons-sketcher-offset-dialog.webp)

The dialog asks for the offset distance and shows a live preview of the
result on the canvas while you type:

- **Closed contours** grow with a positive distance and shrink with a
  negative one. Offsetting past the point where the contour would collapse
  is refused.
- **Open paths** become a closed slot outline of the given width, with
  rounded end caps.

![Bezier contour](/screenshots/addons-sketcher-offset-before.webp)
![Bezier offset into a slot](/screenshots/addons-sketcher-offset-after.webp)

Offsetting replaces the selected contour with the result:

- Lone circles, arcs, and ellipses keep their entity type and are updated
  in place, so they remain editable and constrainable as before.
- Chains of connected segments (including beziers) are replaced by a
  polygon entity. The polygon is edited as a whole: drag its center point
  to move it and the handle point to rotate or uniformly scale it.

If the selection contains several disconnected contours, each one is offset
independently in a single step.

## Chamfer and Fillet

The sketcher provides tools to modify corners of your geometry:

- **Chamfer**: Replaces a sharp corner with a beveled edge. Select a junction
  point (where two lines meet) and apply the chamfer action.
- **Fillet**: Replaces a sharp corner with a rounded edge. Select a junction
  point (where two lines meet) and apply the fillet action.

To use chamfer or fillet:
1. Select a junction point where two lines meet
2. Press `C+H` for chamfer or `C+F` for fillet
3. Use the pie menu or keyboard shortcuts to apply the modification
