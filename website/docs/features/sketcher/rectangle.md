---
description:
  "Draw rectangles and rounded rectangles in the Rayforge sketcher, with center points, modifier
  keys, and dimension input."
---

# Rectangle and Rounded Rectangle

The sketcher offers two rectangle tools that share the same gestures and modifier keys: the
**rectangle** tool (`G+R`) and the **rounded rectangle** tool (`G+O`).

![A rectangle and a rounded rectangle](/screenshots/addons-sketcher-tool-rectangle.webp)

## Drawing Rectangles

Draw a rectangle by specifying two opposite corners, or press at the first corner, drag, and release
at the opposite corner. The modifier keys work the same for both tools:

- Hold `Shift` to place the rectangle symmetrically around the start point.
- Hold `Ctrl` to constrain it to a square.

Each rectangle automatically creates a **center point** constrained to the geometric center, so you
can dimension or snap to the middle of the shape.

While a preview is active, you can type the exact size: the status bar shows the `W` and `H` fields
(plus `R` for the corner radius of rounded rectangles). Type a value, press `Tab` to move between
fields, and `Enter` to apply. Both tools accept the two-click and the click-and-drag gesture
interchangeably; `Esc` cancels the preview.

The rounded rectangle's corner radius can also be changed later by editing its constraints — the
corners are fully constrained, so the radius stays adjustable.
