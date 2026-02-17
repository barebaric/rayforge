# Parametric 2D Sketcher

The Parametric 2D Sketcher is a powerful feature in Rayforge that allows you to create
and edit precise, constraint-based 2D designs directly within the application. This
feature enables you to design custom parts from scratch without needing external CAD
software.

## Overview

The sketcher provides a complete set of tools for creating geometric shapes and applying
parametric constraints to define precise relationships between elements. This approach
ensures your designs maintain their intended geometry even when dimensions are modified.

## Creating and Editing Sketches

### Creating a New Sketch

1. Click the "New Sketch" button in the toolbar or use the main menu
2. A new empty sketch workspace will open with the sketch editor interface
3. Start creating geometry using the drawing tools from the pie menu or keyboard
   shortcuts
4. Apply constraints to define relationships between elements
5. Click "Finish Sketch" to save your work and return to the main workspace

### Editing Existing Sketches

1. Double-click on a sketch-based workpiece in the main workspace
2. Alternatively, select a sketch and choose "Edit Sketch" from the context menu
3. Make your modifications using the same tools and constraints
4. Click "Finish Sketch" to save changes or "Cancel Sketch" to discard them

## Creating 2D Geometry

The sketcher supports creating the following basic geometric elements:

- **Lines**: Draw straight line segments between points
- **Circles**: Create circles by defining a center point and radius
- **Arcs**: Draw arcs by specifying a center point, start point, and end point
- **Rectangles**: Draw rectangles by specifying two opposite corners
- **Rounded Rectangles**: Draw rectangles with rounded corners
- **Text Boxes**: Add text elements to your sketch
- **Fills**: Fill closed regions to create solid areas

These elements form the foundation of your 2D designs and can be combined to create
complex shapes. Fills are particularly useful for creating solid regions that will
be engraved or cut as a single piece.

## Parametric Constraint System

The constraint system is the core of the parametric sketcher, allowing you to define
precise geometric relationships:

### Geometric Constraints

- **Coincident**: Forces two points to occupy the same location
- **Vertical**: Constrains a line to be perfectly vertical
- **Horizontal**: Constrains a line to be perfectly horizontal
- **Tangent**: Makes a line tangent to a circle or arc
- **Perpendicular**: Forces two lines, a line and an arc/circle, or two arcs/circles
  to meet at 90 degrees
- **Point on Line/Shape**: Constrains a point to lie on a line, arc, or circle
- **Symmetry**: Creates symmetrical relationships between elements. Supports two modes:
  - **Point Symmetry**: Select 3 points (first is the center)
  - **Line Symmetry**: Select 2 points and 1 line (the line is the axis)

### Dimensional Constraints

- **Distance**: Sets the exact distance between two points or along a line
- **Diameter**: Defines the diameter of a circle
- **Radius**: Sets the radius of a circle or arc
- **Angle**: Enforces a specific angle between two lines
- **Aspect Ratio**: Forces the ratio between two distances to be equal to a specified
  value
- **Equal Length/Radius**: Forces multiple elements (lines, arcs, or circles) to have
  the same length or radius
- **Equal Distance**: Forces the distance between two point pairs to be equal

## Pie Menu Interface

The sketcher features a context-aware pie menu that provides quick access to all drawing
and constraint tools. This radial menu appears when you right-click in the sketch
workspace and adapts based on your current context and selection.

The pie menu items dynamically show available options based on what you have selected.
For example, when clicking on empty space, you'll see drawing tools. When clicking on
selected geometry, you'll see applicable constraints.

![Sketcher Pie Menu](../images/sketcher-pie-menu.png)

## Keyboard Shortcuts

The sketcher provides keyboard shortcuts for efficient workflow:

### Tool Shortcuts
- `Space`: Select tool
- `G+L`: Line tool
- `G+A`: Arc tool
- `G+C`: Circle tool
- `G+R`: Rectangle tool
- `G+O`: Rounded Rectangle tool
- `G+F`: Fill Area tool
- `G+T`: Text Box tool
- `G+N`: Toggle construction mode on selection

### Action Shortcuts
- `C+H`: Add Chamfer corner
- `C+F`: Add Fillet corner

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
- `Delete`: Delete selected elements
- `Escape`: Cancel current operation or deselect
- `F`: Fit view to content

## Construction Mode

Construction mode allows you to mark entities as "construction geometry" - helper
elements used to guide your design but not part of the final output. Construction
entities are displayed differently (typically as dashed lines) and are not included when
the sketch is used for laser cutting or engraving.

To toggle construction mode:
- Select one or more entities
- Press `N` or `G+N`, or use the Construction option in the pie menu

Construction entities are useful for:
- Creating reference lines and circles
- Defining temporary geometry for alignment
- Building complex shapes from a framework of guides

## Chamfer and Fillet

The sketcher provides tools to modify corners of your geometry:

- **Chamfer**: Replaces a sharp corner with a beveled edge. Select a junction point
  (where two lines meet) and apply the chamfer action.
- **Fillet**: Replaces a sharp corner with a rounded edge. Select a junction point
  (where two lines meet) and apply the fillet action.

To use chamfer or fillet:
1. Select a junction point where two lines meet
2. Press `C+H` for chamfer or `C+F` for fillet
3. Use the pie menu or keyboard shortcuts to apply the modification

## Import and Export

### Exporting Objects

You can export any selected workpiece to various vector formats:

1. Select a workpiece on the canvas
2. Choose **Object → Export Object...** (or right-click and select from context menu)
3. Choose the export format:
   - **RFS (.rfs)**: Rayforge's native parametric sketch format - preserves all
     constraints and can be re-imported for editing
   - **SVG (.svg)**: Standard vector format - widely compatible with design software
   - **DXF (.dxf)**: CAD interchange format - compatible with most CAD applications

### Saving Sketches

You can save your 2D sketches to files for reuse in other projects. All parametric
constraints are preserved when saving, ensuring your designs maintain their geometric
relationships.

### Importing Sketches

Saved sketches can be imported into any workspace, allowing you to create a library of
commonly used design elements. The import process maintains all constraints and
dimensional relationships.

## Workflow Tips

1. **Start with Rough Geometry**: Create basic shapes first, then refine with constraints
2. **Use Constraints Early**: Apply constraints as you build to maintain design intent
3. **Check Constraint Status**: The system indicates when sketches are fully constrained
4. **Watch for Conflicts**: Constraints that conflict with each other are highlighted in red
5. **Utilize Symmetry**: Symmetry constraints can significantly speed up complex designs
6. **Iterate and Refine**: Don't hesitate to modify constraints to achieve the desired
   result

## Editing Features

- **Full Undo/Redo Support**: The entire sketch state is saved with each operation
- **Dynamic Cursor**: The cursor changes to reflect the active drawing tool
- **Constraint Visualization**: Applied constraints are clearly indicated in the interface
- **Real-time Updates**: Changes to constraints immediately update the geometry
- **Double-Click Editing**: Double-click on dimensional constraints (Distance, Radius, Diameter,
  Angle, Aspect Ratio) opens a dialog to edit their values
- **Parametric Expressions**: Dimensional constraints support expressions, allowing values to be
  calculated from other parameters (e.g., `width/2` for a radius that's half the
  width)
