---
description: "Learn how to create lines, bezier curves, arcs, ellipses, rectangles, and other 2D geometry in the Rayforge sketcher."
---

# Creating 2D Geometry

The sketcher supports creating the following basic geometric elements:

- **Paths (Lines and Bezier Curves)**: Draw straight lines and smooth bezier
  curves using the unified path tool. Click to place points, drag to create
  bezier handles.
- **Arcs**: Draw arcs by specifying a center point, start point, and end point.
- **Ellipses**: Create ellipses (and circles) with two clicks: the first sets
  the center, the second sets the edge point. You can also press at the
  center, drag, and release at the edge - both gestures work
  interchangeably. Hold `Ctrl` to constrain to a perfect circle and `Shift`
  to use the start point as the ellipse's center.
- **Rectangles**: Draw rectangles by specifying two opposite corners, or
  press at the first corner, drag, and release at the opposite corner. Each
  rectangle auto-creates a center point (constrained to the geometric
  center) so you can dimension or snap to it. Hold `Shift` while drawing to
  place the rectangle symmetrically around the start point, and `Ctrl` to
  constrain it to a square.
- **Rounded Rectangles**: Draw rectangles with rounded corners using the
  same gestures and modifiers as the rectangle tool: two clicks or
  click-and-drag, with `Shift` to center on the start point and `Ctrl` to
  constrain to a square. The corner radius can be set by typing dimensions
  (`0-9`, fields W, H and R).
- **Text Boxes**: Add text elements to your sketch. Text content supports
  parametric template expressions (see [Text Templates](../text.md)).
- **Fills**: Fill closed regions to create solid areas

These elements form the foundation of your 2D designs and can be combined to
create complex shapes. Fills are particularly useful for creating solid regions
that will be engraved or cut as a single piece.

## Two-Click or Drag

The shape-creation tools (ellipse, rectangle, rounded rectangle) accept two
gestures interchangeably: click the first point, move, and click the second
point, or press at the first point, drag, and release at the second. A quick
click without movement simply arms the tool and waits for the second point,
so stray clicks never leave degenerate geometry behind. While a preview is
active, the status bar shows the available modifier keys, and `Esc` cancels
the preview.

## Working with Bezier Curves

The path tool supports bezier curves for creating smooth, organic shapes:

### Drawing Bezier Curves

1. Select the path tool from the pie menu or use the keyboard shortcut
2. Click to place points - each click creates a new point
3. Drag after clicking to create bezier handles for smooth curves
4. Continue adding points to build your path
5. Press Escape or double-click to finish the path

### Editing Bezier Curves

- **Move points**: Click and drag any point to reposition it
- **Adjust handles**: Drag the handle endpoints to modify the curve shape
- **Connect to existing points**: When editing a path, you can snap to existing
  points in your sketch
- **Make smooth/symmetric**: Points connected by a coincident constraint can be
  made smooth (continuous tangent) or symmetric (mirrored handles)

### Converting Curves to Lines

Use the **straighten tool** to convert bezier curves back to straight lines.
This is useful when you need clean, simple geometry. Select the bezier segments
you want to convert and apply the straighten action.
