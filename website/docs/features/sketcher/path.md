---
description: "Draw straight lines and smooth bezier curves with the path tool in the Rayforge sketcher."
---

# Path Tool

The path tool (`G+P` or `G+L`) draws connected chains of straight lines
and smooth bezier curves in one unified workflow. It is the most
versatile drawing tool in the sketcher: click to place points, drag to
bend the segment into a curve.

![A path of two lines joined by a bezier segment, with its waypoints and handles](/screenshots/addons-sketcher-tool-path.webp)

## Drawing Paths

1. Select the path tool from the pie menu, the **Sketch** menu, or with
   `G+P`.
2. Click to place the first point. A live preview follows the cursor.
3. Click again without dragging to finish a straight segment — the next
   segment immediately starts from that point.
4. Press at a point and drag before releasing to turn the segment into
   a bezier curve. The drag controls the "bow" of the curve.
5. Keep adding points to build your path.
6. Press `Escape` or double-click to finish the path.

While a preview is active, the status bar lists the modifier keys that
apply, and `Esc` cancels it.

## Working with Bezier Curves

Bezier curves create smooth, organic shapes:

- **Adjust handles**: select a bezier and drag the round handle
  endpoints to modify the curve shape. Each handle bends the curve on
  its side of the waypoint.
- **Connect to existing points**: while drawing, magnetic snapping
  attaches new segments to existing points in your sketch, and the
  matching constraint is created automatically.

### Waypoint Types

The point where two segments of a path meet is a *waypoint*. The
waypoint type controls how the curve flows through it:

- **Sharp**: the handles on both sides are independent, producing a
  corner.
- **Smooth**: the handles share a tangent, producing a continuous,
  rounded transition.
- **Symmetric**: like Smooth, but the handles are also mirrored, so
  both sides bend equally.

To change a waypoint's type, right-click it (or the adjoining bezier
segment) and pick the type from the pie menu. Newly drawn bezier
waypoints are symmetric.

![The pie menu on a selected bezier waypoint, with the Straighten, Sharp, Smooth, and Symmetric tools](/screenshots/addons-sketcher-tool-path-pie-menu.webp)

### Converting Curves to Lines

The **Straighten** tool from the same pie menu converts bezier curves
back to straight lines, which is useful when you need clean, simple
geometry. Select the bezier segments you want to convert and apply the
straighten action. The segments collapse to the straight connection
between their endpoints.

## Automatic Constraints

The path tool participates in magnetic snapping like every other
drawing tool. When snap guides show alignment during drawing, matching
horizontal and vertical constraints are created automatically, which
keeps your sketch tidy from the start rather than fixing things up
afterward. Hold `Shift` to constrain the new segment to the nearest
axis. See [Grid and snapping](index.md#grid-and-snapping) for the full
list of snap indicators.
