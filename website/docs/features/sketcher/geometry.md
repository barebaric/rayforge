---
description: "Learn how to create lines, bezier curves, arcs, ellipses, rectangles, and other 2D geometry in the Rayforge sketcher."
---

# Creating 2D Geometry

The sketcher supports creating the following basic geometric elements:

- **Paths (Lines and Bezier Curves)**: Draw straight lines and smooth bezier
  curves using the unified path tool. Click to place points, drag to create
  bezier handles.
- **Arcs**: Draw arcs by specifying a center point, start point, and end point
- **Ellipses**: Create ellipses (and circles) by defining a center point and
  dragging to set the size and aspect ratio. Hold `Ctrl` while dragging to
  constrain to a perfect circle.
- **Rectangles**: Draw rectangles by specifying two opposite corners.
  Each rectangle auto-creates a center point (constrained to the geometric
  center) so you can dimension or snap to it. Hold `Shift` while drawing to
  place the rectangle symmetrically around the start point, matching the
  ellipse tool.
- **Rounded Rectangles**: Draw rectangles with rounded corners
- **Text Boxes**: Add text elements to your sketch. Text content supports
  parametric template expressions (see [Text Templates](../text.md)).
- **Fills**: Fill closed regions to create solid areas

These elements form the foundation of your 2D designs and can be combined to
create complex shapes. Fills are particularly useful for creating solid regions
that will be engraved or cut as a single piece.

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
