---
description: "Use Rayforge's built-in parametric 2D sketcher to create custom laser-ready designs with lines, circles, bezier curves, and constraints."
---

# Parametric 2D Sketcher

The Parametric 2D Sketcher is a powerful feature in Rayforge that allows you to
create and edit precise, constraint-based 2D designs directly within the
application. This feature enables you to design custom parts from scratch
without needing external CAD software.

## Overview

The sketcher provides a complete set of tools for creating geometric shapes and
applying parametric constraints to define precise relationships between elements.
This approach ensures your designs maintain their intended geometry even when
dimensions are modified.

## Creating and Editing Sketches

### Creating a New Sketch

1. Open the bottom panel and click the **New Sketch** button, or right-click
   on the canvas and select **New Sketch** from the context menu.
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

## Workflow Tips

1. **Start with Rough Geometry**: Create basic shapes first, then refine with
   constraints
2. **Use Constraints Early**: Apply constraints as you build to maintain design
   intent
3. **Check Constraint Status**: The system indicates when sketches are fully
   constrained
4. **Watch for Conflicts**: Constraints that conflict with each other are
   highlighted in red and shown in the constraints panel for easy identification
5. **Utilize Symmetry**: Symmetry constraints can significantly speed up complex
   designs
6. **Use the Grid**: Enable the grid for precise alignment, and use Ctrl to snap
   to grid
7. **Iterate and Refine**: Don't hesitate to modify constraints to achieve the
   desired result

## Editing Features

- **Full Undo/Redo Support**: The entire sketch state is saved with each
  operation
- **Dynamic Cursor**: The cursor changes to reflect the active drawing tool
- **Constraint Visualization**: Applied constraints are clearly indicated in the
  interface
- **Real-time Updates**: Changes to constraints immediately update the geometry
- **Double-Click Editing**: Double-click on dimensional constraints (Distance,
  Radius, Diameter, Angle, Aspect Ratio) opens a dialog to edit their values
- **Parametric Expressions**: Dimensional constraints support expressions,
  allowing values to be calculated from other parameters (e.g., `width/2` for a
  radius that's half the width)
