---
description:
  "Draw arcs and ellipses (including circles) in the Rayforge sketcher, with keyboard modifiers and
  dimension input."
---

# Arc and Ellipse

The sketcher provides two curved-shape tools: the **arc tool** for circular arcs and the **ellipse
tool** for ellipses and circles.

![An arc and an ellipse as created by their tools](/screenshots/addons-sketcher-tool-arc-ellipse.webp)

## Arc Tool

The arc tool (`G+A`) creates an arc in three clicks:

1. Click the **center** point.
2. Click the **start** point — its distance from the center sets the radius.
3. Move the cursor to preview the arc sweeping between the two points and click the **end**
   position.

While the preview is active, you can type a number to fix the radius exactly; press `Tab` or `Enter`
to apply it. `Tab` before typing toggles magnetic snapping.

## Ellipse Tool

The ellipse tool (`G+C`) creates ellipses and circles with two clicks: the first sets the center,
the second sets the edge point. You can also press at the center, drag, and release at the edge —
both gestures work interchangeably.

- Hold `Ctrl` to constrain the shape to a perfect circle.
- Hold `Shift` to use the start point as the ellipse's center.

## Two-Click or Drag

Like the [rectangle](rectangle.md) tools, the ellipse tool accepts two gestures interchangeably:
click the first point, move, and click the second point, or press at the first point, drag, and
release at the second. A quick click without movement simply arms the tool and waits for the second
point, so stray clicks never leave degenerate geometry behind. While a preview is active, the status
bar shows the available modifier keys, and `Esc` cancels the preview.
